import os
import json
import re
from typing import List, Dict, Any
from dotenv import load_dotenv
import pandas as pd
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from .prompts import SYSTEM_PROMPT
from .models import Plan
import logging

logger = logging.getLogger("uvicorn.error")

load_dotenv()

# ---------- Config ----------
DEFAULT_BASE = "https://integrate.api.nvidia.com/v1"
BASE_URL = os.getenv("LLM_BASE_URL", DEFAULT_BASE).rstrip("/")
API_KEY = (
    os.getenv("LLM_API_KEY") or os.getenv("NVIDIA_API_KEY") or os.getenv("NVCF_API_KEY")
)
MODEL = os.getenv("LLM_MODEL", "openai/gpt-oss-120b")


class LLMError(RuntimeError):
    pass


def _get_llm(temperature: float = 0.1) -> ChatNVIDIA:
    if not API_KEY:
        raise LLMError(
            "LLM API key missing. Set LLM_API_KEY (or NVIDIA_API_KEY / NVCF_API_KEY) in backend/.env"
        )
    return ChatNVIDIA(
        model=MODEL,
        temperature=temperature,
        base_url=BASE_URL,
        api_key=API_KEY,
    )


# ---------- helpers for prompt construction ----------


def _as_text_from_content(content: Any) -> str:
    """Normalize LC content (str | list[chunk] | dict | AIMessage)."""
    if content is None:
        return ""
    if isinstance(content, AIMessage):
        return _as_text_from_content(content.content)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for p in content:
            if isinstance(p, str):
                parts.append(p)
            elif isinstance(p, dict):
                t = p.get("text")
                if isinstance(t, str):
                    parts.append(t)
                else:
                    parts.append(str(p))
            else:
                parts.append(str(p))
        return "".join(parts)
    if isinstance(content, dict):
        if isinstance(content.get("text"), str):
            return content["text"]
        if isinstance(content.get("content"), str):
            return content["content"]
        return str(content)
    return str(content)


def _coerce_plan_object(obj: Any) -> Dict[str, Any]:
    """
    Ensure we return a single dict plan. Accepts:
    - dict -> as-is
    - list -> pick first dict with 'action' or the first dict
    - str  -> try json.loads and recurse
    Raises LLMError if no usable dict is found.
    """
    if isinstance(obj, dict):
        return obj
    if isinstance(obj, list):
        # prefer an item that looks like a plan
        for it in obj:
            if isinstance(it, dict) and (
                "action" in it or "type" in it or "vega_lite" in it
            ):
                return it
        # fallback: first dict
        for it in obj:
            if isinstance(it, dict):
                return it
        raise LLMError("plan_not_object: array contained no dict plans")
    if isinstance(obj, str):
        try:
            import json

            return _coerce_plan_object(json.loads(obj))
        except Exception as e:
            raise LLMError(f"plan_not_object: string not JSON: {e}")
    raise LLMError(f"plan_not_object: unsupported type {type(obj).__name__}")


def _as_text_from_response(resp: Any) -> str:
    """
    Try multiple places providers may stash text:
      - resp.content (usual)
      - resp.additional_kwargs.reasoning_content (NVIDIA)
      - resp.additional_kwargs.content
      - resp.message.content (rare dict shape)
    """
    # 1) normal path
    text = _as_text_from_content(getattr(resp, "content", None))
    if text:
        return text

    # 2) provider-specific extras
    extras = getattr(resp, "additional_kwargs", {}) or {}
    if isinstance(extras, dict):
        rc = extras.get("reasoning_content")
        if isinstance(rc, str) and rc.strip():
            return rc
        c2 = extras.get("content")
        if isinstance(c2, str) and c2.strip():
            return c2
        msg = extras.get("message")
        if isinstance(msg, dict):
            mc = msg.get("content")
            if isinstance(mc, str) and mc.strip():
                return mc

    # 3) raw dict
    if isinstance(resp, dict):
        c = resp.get("content")
        if isinstance(c, str) and c.strip():
            return c
        extras = resp.get("additional_kwargs") or {}
        if isinstance(extras, dict):
            rc = extras.get("reasoning_content")
            if isinstance(rc, str) and rc.strip():
                return rc

    return ""


def columns_markdown(df: pd.DataFrame) -> str:
    return "\n".join(f"- {c} ({str(df[c].dtype)})" for c in df.columns)


def build_user_prompt(
    columns_markdown: str,
    user_prompt: str,
    client_ctx: Dict[str, Any] | None = None,
    *,
    profile: Dict[str, Any] | None = None,
) -> str:
    ctx_txt = ""
    if client_ctx:
        cards = client_ctx.get("cards") or []
        selected = client_ctx.get("selection")
        if cards:
            lines = [f"- {c.get('id')} :: {c.get('title')}" for c in cards]
            ctx_txt = "Existing visualizations:\n" + "\n".join(lines)
        if selected:
            ctx_txt += ("\n" if ctx_txt else "") + f"Selected: {selected}"

    profile_txt = ""
    if profile:
        # Keep it LLM-friendly: short JSON-ish block
        # (Don’t dump the whole DF; just this summary.)
        profile_txt = (
            "Dataset profile (compact JSON):\n"
            + json.dumps(profile, ensure_ascii=False)[:8000]
        )

    return f"""Dataset columns:
{columns_markdown}

{profile_txt if profile_txt else ""}

{ctx_txt if ctx_txt else ""}

User prompt:
{user_prompt}

Return ONLY JSON as specified by the system prompt. Do not include any explanatory text or formatting fences.
"""


def _extract_json_block(text: str) -> str:
    if not text:
        raise ValueError("empty LLM response")
    # strip common fences
    text = re.sub(
        r"^```(?:json)?\s*|\s*```$",
        "",
        text.strip(),
        flags=re.IGNORECASE | re.MULTILINE,
    )
    # quick path
    try:
        json.loads(text)
        return text
    except Exception:
        pass

    # scan for the first balanced JSON object/array
    start = None
    depth = 0
    for i, ch in enumerate(text):
        if ch in "{[":
            start = i
            depth = 1
            break
    if start is None:
        raise ValueError("no JSON start in response")

    for j in range(start + 1, len(text)):
        if text[j] in "{[":
            depth += 1
        elif text[j] in "}]":
            depth -= 1
            if depth == 0:
                candidate = text[start : j + 1]
                json.loads(candidate)
                return candidate

    teaser = text[start : start + 400].replace("\n", "\\n")
    raise ValueError(f"unterminated JSON (teaser): {teaser}")


def _strip_code_fences(text: str) -> str:
    return re.sub(
        r"^```(?:json)?\s*|\s*```$",
        "",
        text.strip(),
        flags=re.IGNORECASE | re.MULTILINE,
    )


def _first_balanced_json(text: str) -> str:
    text = _strip_code_fences(text)
    # fast path
    try:
        json.loads(text)
        return text
    except Exception:
        pass
    start = None
    depth = 0
    for i, ch in enumerate(text):
        if ch in "{[":
            start = i
            depth = 1
            break
    if start is None:
        raise ValueError("no JSON start in response")
    for j in range(start + 1, len(text)):
        if text[j] in "{[":
            depth += 1
        elif text[j] in "}]":
            depth -= 1
            if depth == 0:
                return text[start : j + 1]
    teaser = text[start : start + 400].replace("\n", "\\n")
    raise ValueError(f"unterminated JSON (teaser): {teaser}")


_re_trailing_commas = re.compile(r",(\s*[}\]])")
_re_single_quoted = re.compile(r"'([^'\\]*(?:\\.[^'\\]*)*)'")
_re_bare_literals = re.compile(r"\b(?:None|True|False)\b")


def _try_repair_json(s: str) -> dict:
    t = _re_trailing_commas.sub(r"\1", s)
    t = _re_bare_literals.sub(
        lambda m: {"None": "null", "True": "true", "False": "false"}[m.group(0)], t
    )
    t = _re_single_quoted.sub(lambda m: '"' + m.group(1).replace('"', '\\"') + '"', t)
    return json.loads(t)


def _load_plan_json(text: str) -> dict:
    txt = _strip_code_fences(text or "")
    if not txt.strip():
        raise ValueError("empty LLM response text")
    try:
        return json.loads(txt)
    except Exception:
        pass
    block = _first_balanced_json(txt)
    try:
        return json.loads(block)
    except Exception:
        try:
            return _try_repair_json(block)
        except Exception as e:
            teaser = block[:400].replace("\n", "\\n")
            raise ValueError(f"json_parse_failed after repair: {e}; teaser={teaser}")


def chat_plan_structured(system_prompt: str, user_message: str) -> Plan:
    llm = _get_llm()
    try:
        llm_struct = llm.with_structured_output(Plan)
        msg = [SystemMessage(system_prompt), HumanMessage(user_message)]
        out = llm_struct.invoke(msg)
        if out is None:
            raise LLMError("structured_output_failed: empty result")
        if isinstance(out, Plan):
            return out
        if isinstance(out, dict):
            return Plan(**out)
        # last chance: parse text
        text = _as_text_from_response(out) or _as_text_from_content(
            getattr(out, "content", None)
        )
        block = _extract_json_block(text)
        obj = json.loads(block)
        obj = _coerce_plan_object(obj)
        return Plan(**obj)
    except Exception as e:
        raise LLMError(f"structured_output_failed: {e}")


def chat_json(system_prompt: str, user_message: str) -> dict:
    llm = _get_llm().bind(
        extra_body={
            "response_format": {"type": "json_object"},
            "max_tokens": 512,
            # "temperature": 0.1,  # optional
        }
    )
    messages = [
        SystemMessage(
            system_prompt + "\nReturn ONE JSON object. No prose, no code fences."
        ),
        HumanMessage(user_message),
    ]
    resp = llm.invoke(messages)
    text = _as_text_from_response(resp) or _as_text_from_content(
        getattr(resp, "content", None)
    )
    if not text or not text.strip():
        extras = getattr(resp, "additional_kwargs", None)
        raise LLMError(f"no_content: additional={extras}")
    # DEBUG (optional): log a tiny teaser to see what came back
    logger.debug("LLM raw text teaser: %r", text[:200])
    obj = _load_plan_json(text)
    return _coerce_plan_object(obj)


def plan_from_llm(
    columns_md: str,
    user_prompt: str,
    client_ctx: dict | None,
    *,
    profile: dict | None = None,
) -> dict:
    user_msg = build_user_prompt(columns_md, user_prompt, client_ctx, profile=profile)
    return chat_json(SYSTEM_PROMPT, user_msg)

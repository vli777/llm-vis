import os, json, re
from typing import List, Optional, Literal, Dict, Any, Union
from dotenv import load_dotenv
from pydantic import BaseModel, Field
import pandas as pd
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from .prompts import SYSTEM_PROMPT
from .models import Plan, Operation

load_dotenv()

# ---------- Config ----------
DEFAULT_BASE = "https://integrate.api.nvidia.com/v1"
BASE_URL = os.getenv("LLM_BASE_URL", DEFAULT_BASE).rstrip("/")
API_KEY = os.getenv("LLM_API_KEY") or os.getenv("NVIDIA_API_KEY") or os.getenv("NVCF_API_KEY")
MODEL = os.getenv("LLM_MODEL", "openai/gpt-oss-120b")

class LLMError(RuntimeError):
    pass

def _get_llm(temperature: float = 0.2) -> ChatNVIDIA:
    if not API_KEY:
        raise LLMError("LLM API key missing. Set LLM_API_KEY (or NVIDIA_API_KEY / NVCF_API_KEY) in backend/.env")
    return ChatNVIDIA(
        model=MODEL,
        temperature=temperature,
        base_url=BASE_URL,
        api_key=API_KEY,
    )

# ---------- helpers for prompt construction ----------
# llm.py
from typing import Any, Dict, List
from langchain_core.messages import AIMessage

def _as_text_from_content(content: Any) -> str:
    """Normalize LC content (str | list[chunk] | dict)."""
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
        # common shapes
        if "text" in content and isinstance(content["text"], str):
            return content["text"]
        if "content" in content and isinstance(content["content"], str):
            return content["content"]
        return str(content)
    return str(content)

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

def build_user_prompt(columns_markdown: str, user_prompt: str, client_ctx: Dict[str, Any] | None = None, *, profile: Dict[str, Any] | None = None) -> str:
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
        profile_txt = "Dataset profile (compact JSON):\n" + json.dumps(profile, ensure_ascii=False)[:8000]

    return f"""Dataset columns:
{columns_markdown}

{profile_txt if profile_txt else ""}

{ctx_txt if ctx_txt else ""}

User prompt:
{user_prompt}

Return ONLY JSON as specified by the system prompt. Do not include any explanatory text or formatting fences.
"""

def _as_text_from_content(content: Any) -> str:
    """
    Normalize LangChain provider response content to plain text.
    Handles:
      - str
      - list[dict|str] with text chunks (OpenAI/NVIDIA style)
      - AIMessage (uses .content and .additional_kwargs)
    """
    if content is None:
        return ""

    # If it's the newer LC message type
    if isinstance(content, AIMessage):
        return _as_text_from_content(content.content)

    # If the provider returned plain text
    if isinstance(content, str):
        return content

    # If it's a list of chunks
    if isinstance(content, list):
        parts: List[str] = []
        for p in content:
            if isinstance(p, str):
                parts.append(p)
            elif isinstance(p, dict):
                # Common shapes: {'type':'text','text':'...'} or {'text':'...'}
                t = p.get("text")
                if isinstance(t, str):
                    parts.append(t)
                else:
                    # fallback: stringify but keep it short
                    parts.append(str(p))
            else:
                parts.append(str(p))
        return "".join(parts)

    # Fallback: stringify (last resort)
    return str(content)

def _extract_json_block(text: str) -> str:
    if not text:
        raise ValueError("empty LLM response")
    # strip common fences
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.IGNORECASE | re.MULTILINE)
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
                candidate = text[start:j + 1]
                json.loads(candidate)
                return candidate

    teaser = text[start:start+400].replace("\n", "\\n")
    raise ValueError(f"unterminated JSON (teaser): {teaser}")

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

        # Last chance: try to parse text out of the raw message
        text = _as_text_from_response(out) or _as_text_from_content(getattr(out, "content", None))
        if not text:
            raise LLMError("structured_output_failed: no parsed text")
        block = _extract_json_block(text)
        return Plan(**json.loads(block))
    except Exception as e:
        raise LLMError(f"structured_output_failed: {e}")

def chat_json(system_prompt: str, user_message: str) -> Dict[str, Any]:
    llm = _get_llm()
    messages = [
        SystemMessage(system_prompt + "\nOutput ONLY JSON. No prose. Start with '{' and return a single JSON object."),
        HumanMessage(user_message),
    ]

    def invoke_once(tighter: bool = False) -> str:
        try:
            if tighter:
                llm_json = llm.bind(extra_body={"response_format": {"type": "json_object"}})
                resp = llm_json.invoke(messages)
            else:
                resp = llm.invoke(messages)
            text = _as_text_from_response(resp)
            if not text:
                # fall back to content normalization (older providers)
                text = _as_text_from_content(getattr(resp, "content", None))
            if not text:
                raw = getattr(resp, "additional_kwargs", None)
                raise LLMError(f"no_content: additional={raw}")
            return text
        except Exception as ex:
            raise LLMError(f"invoke_failed: {ex}")

    content = invoke_once(tighter=False)
    try:
        block = _extract_json_block(content)
        return json.loads(block)
    except Exception:
        content2 = invoke_once(tighter=True)
        block2 = _extract_json_block(content2)  # let this raise with teaser
        return json.loads(block2)

def plan_from_llm(columns_markdown: str, user_prompt: str, client_ctx: Dict[str, Any] | None, *, profile: Dict[str, Any] | None = None) -> Dict[str, Any]:
    user_msg = build_user_prompt(columns_markdown, user_prompt, client_ctx, profile=profile)
    try:
        plan = chat_plan_structured(SYSTEM_PROMPT, user_msg)
        return plan.model_dump(by_alias=True)
    except Exception as e_struct:
        try:
            return chat_json(SYSTEM_PROMPT, user_msg)
        except Exception as e_json:
            raise LLMError(f"LLM failed: {e_struct}; fallback failed: {e_json}")
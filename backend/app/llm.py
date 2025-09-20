import os, json, re
from typing import List, Optional, Literal, Dict, Any
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_core.messages import SystemMessage, HumanMessage
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

def build_user_prompt(columns_markdown: str, user_prompt: str, client_ctx: Dict[str, Any] | None = None) -> str:
    """Compose the user message with schema + optional client context (cards, selection)."""
    ctx_txt = ""
    if client_ctx:
        cards = client_ctx.get("cards") or []
        selected = client_ctx.get("selection")
        if cards:
            lines = [f"- {c.get('id')} :: {c.get('title')}" for c in cards]
            ctx_txt = "Existing visualizations:\n" + "\n".join(lines)
        if selected:
            ctx_txt += ("\n" if ctx_txt else "") + f"Selected: {selected}"
    return f"""Dataset columns:
{columns_markdown}

{ctx_txt if ctx_txt else ""}

User prompt:
{user_prompt}

Return ONLY JSON with the keys described by the system prompt."""

# ---------- robust JSON extraction ----------

def _extract_json_block(text: str) -> str:
    if not text:
        raise ValueError("empty LLM response")
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.IGNORECASE | re.MULTILINE)
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
        raise ValueError("no JSON start")
    for j in range(start + 1, len(text)):
        if text[j] in "{[":
            depth += 1
        elif text[j] in "}]":
            depth -= 1
            if depth == 0:
                candidate = text[start:j+1]
                json.loads(candidate)
                return candidate
    raise ValueError("unterminated JSON")


# ---------- LLM invocations ----------

def chat_plan_structured(system_prompt: str, user_message: str) -> Plan:
    """Use LangChain structured output (Pydantic) for robust JSON."""
    llm = _get_llm()
    try:
        llm_struct = llm.with_structured_output(Plan)
        msg = [SystemMessage(system_prompt), HumanMessage(user_message)]
        return llm_struct.invoke(msg)
    except Exception as e:
        raise LLMError(f"structured_output_failed: {e}")

def chat_json(system_prompt: str, user_message: str) -> Dict[str, Any]:
    """If structured output isn't supported, parse JSON manually."""
    llm = _get_llm()
    messages = [
        SystemMessage(system_prompt + "\nOutput ONLY JSON. No prose."),
        HumanMessage(user_message),
    ]
    resp = llm.invoke(messages)
    content = resp.content if isinstance(resp.content, str) else str(resp.content)
    try:
        block = _extract_json_block(content)
        return json.loads(block)
    except Exception as e:
        raise LLMError(f"json_parse_failed: {e}; raw_head={content[:200]}")

def plan_from_llm(columns_markdown: str, user_prompt: str, client_ctx: Dict[str, Any] | None) -> Dict[str, Any]:
    """Top-level planner used by nlq_llm.handle_llm_nlq."""
    user_msg = build_user_prompt(columns_markdown, user_prompt, client_ctx)
    try:
        plan = chat_plan_structured(SYSTEM_PROMPT, user_msg)
        return plan.model_dump(by_alias=True)
    except Exception:
        return chat_json(SYSTEM_PROMPT, user_msg)
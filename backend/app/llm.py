import os, json, re
from typing import List, Optional, Literal, Dict, Any
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_core.messages import SystemMessage, HumanMessage

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

# ----- Pydantic schema for structured output (preferred) -----
class Operation(BaseModel):
    op: str
    col: Optional[str] = None
    x: Optional[str] = None
    y: Optional[str] = None
    sep: Optional[str] = None
    as_: Optional[List[str]] = Field(default=None, alias="as")
    extras: Optional[List[str]] = None
    log: Optional[bool] = None

class Plan(BaseModel):
    intent: Literal["chart", "table"]
    title: str
    operations: List[Operation] = Field(default_factory=list)
    vega_lite: Optional[Dict[str, Any]] = None

SYSTEM_INSTR = """You generate data-viz plans as JSON for a backend executor.
Rules:
- Use ONLY the provided dataset columns; never invent columns.
- intent is "chart" or "table".
- vega_lite must be valid Vega-Lite v5 JSON if intent="chart"; set null for tables.
- Data is computed in the backend via 'operations'. Supported ops:
  - value_counts: {op:"value_counts", col:<string>, as:[<key_col>,"n"]}
  - explode_counts: {op:"explode_counts", col:<string>, sep:<string>, as:[<key_col>,"n"]}
  - scatter_data: {op:"scatter_data", x:<string>, y:<string>, extras:[<string>...], log:<bool>}
  - corr_pair: {op:"corr_pair", x:<string>, y:<string>}
Output must match the schema exactly.
"""

def build_user_prompt(columns_markdown: str, user_prompt: str) -> str:
    return f"""Dataset columns:
{columns_markdown}

User prompt:
{user_prompt}

Return a plan with keys: intent, title, operations, vega_lite.
"""

def _extract_json_block(text: str) -> str:
    """Grab the first top-level {...} or [...] block (handles code fences / prefaces)."""
    if not text:
        raise ValueError("empty LLM response")
    # strip code fences if present
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.IGNORECASE | re.MULTILINE)
    # try direct parse
    try:
        json.loads(text)
        return text
    except Exception:
        pass
    # fallback: find first JSON object/array by bracket matching
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
                json.loads(candidate)  # will raise if invalid
                return candidate
    raise ValueError("unterminated JSON")

def chat_plan_structured(columns_markdown: str, user_prompt: str) -> Plan:
    """Use LangChain structured output (Pydantic) for robust JSON."""
    llm = _get_llm()
    try:
        llm_struct = llm.with_structured_output(Plan)
        msg = [
            SystemMessage(SYSTEM_INSTR), 
            HumanMessage(build_user_prompt(columns_markdown, user_prompt))
        ]
        return llm_struct.invoke(msg)
    except Exception as e:
        # Some models may not support structured output; let caller fall back
        raise LLMError(f"structured_output_failed: {e}")

# ---------- JSON fallback ----------
def chat_json(columns_markdown: str, user_prompt: str) -> Dict[str, Any]:
    """If structured output isn't supported by the model, parse JSON manually."""
    llm = _get_llm()
    messages = [
        SystemMessage(SYSTEM_INSTR + "\nOutput ONLY JSON. No prose."),
        HumanMessage(build_user_prompt(columns_markdown, user_prompt)),
    ]
    resp = llm.invoke(messages)
    content = resp.content if isinstance(resp.content, str) else str(resp.content)
    try:
        block = _extract_json_block(content)
        return json.loads(block)
    except Exception as e:
        raise LLMError(f"json_parse_failed: {e}; raw_head={content[:200]}")

def plan_from_llm(columns_markdown: str, user_prompt: str) -> Dict[str, Any]:
    """
    Returns a dict plan matching Plan model (for your executor).
    Tries structured output first; falls back to JSON parsing.
    """
    try:
        plan = chat_plan_structured(columns_markdown, user_prompt)
        return plan.model_dump(by_alias=True)
    except Exception:
        return chat_json(columns_markdown, user_prompt)

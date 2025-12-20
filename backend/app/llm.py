import json
import re
from typing import List, Dict, Any
import pandas as pd
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from .prompts import SYSTEM_PROMPT
from .models import Plan
from .llm_loader import get_chat_model, LLMConfigError, supports_native_structured_output, get_provider_name
import logging

logger = logging.getLogger("uvicorn.error")


class LLMError(RuntimeError):
    pass


def _get_llm(temperature: float = 0.1):
    try:
        return get_chat_model(temperature=temperature)
    except LLMConfigError as exc:
        raise LLMError(str(exc)) from exc


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
    """
    Build a rich, context-aware prompt for the LLM.

    Includes:
    - Column schema with type information
    - Dataset profile with statistics and examples
    - Visualization hints and suggestions
    - Existing visualization context
    - User's natural language query
    """

    # Existing visualizations context
    ctx_txt = ""
    if client_ctx:
        cards = client_ctx.get("cards") or []
        selected = client_ctx.get("selection")
        if cards:
            lines = [f"- {c.get('id')} :: {c.get('title')}" for c in cards]
            ctx_txt = "Existing visualizations:\n" + "\n".join(lines)
        if selected:
            ctx_txt += ("\n" if ctx_txt else "") + f"Selected: {selected}"

    # Enhanced dataset profile
    profile_txt = ""
    viz_hints_txt = ""

    if profile:
        # Extract visualization hints if available
        viz_hints = profile.get("visualization_hints", {})
        if viz_hints:
            suggestions = viz_hints.get("suggested_chart_types", [])
            summary = viz_hints.get("summary", {})

            viz_hints_txt = "VISUALIZATION GUIDANCE:\n"
            viz_hints_txt += f"- Dataset has {summary.get('numeric_columns', 0)} numeric columns, "
            viz_hints_txt += f"{summary.get('categorical_columns', 0)} categorical columns\n"

            if summary.get('has_temporal_data'):
                viz_hints_txt += "- Contains temporal/time-series data\n"

            if suggestions:
                viz_hints_txt += f"- Suggested chart types: {', '.join(suggestions)}\n"

        # Compact profile for token efficiency but still informative
        # Include column details with roles and sample values
        profile_compact = {
            "row_count": profile.get("row_count"),
            "columns": []
        }

        for col in profile.get("columns", []):
            col_info = {
                "name": col.get("name"),
                "dtype": col.get("dtype"),
                "role": col.get("role", "unknown"),
                "unique": col.get("unique"),
                "missing_pct": col.get("missing_pct", 0),
            }

            # Add relevant stats based on column type
            if "num_stats" in col:
                stats = col["num_stats"]
                col_info["range"] = f"{stats.get('min', 'N/A')} to {stats.get('max', 'N/A')}"
                col_info["mean"] = stats.get("mean")

            elif "datetime_range" in col:
                col_info["time_range"] = col["datetime_range"]

            elif "top_values" in col:
                # Show top 3 values for categorical
                top_vals = col["top_values"][:3]
                col_info["top_values"] = [f"{v['value']} ({v['n']})" for v in top_vals]

            # Always include examples
            col_info["examples"] = col.get("examples", [])

            profile_compact["columns"].append(col_info)

        # Add sample rows
        if "sample_rows" in profile:
            profile_compact["sample_rows"] = profile["sample_rows"]

        profile_txt = (
            "DATASET PROFILE:\n"
            + json.dumps(profile_compact, ensure_ascii=False, indent=2)[:6000]
        )

    return f"""DATASET SCHEMA:
{columns_markdown}

{profile_txt}

{viz_hints_txt}

{ctx_txt if ctx_txt else ""}

USER REQUEST:
{user_prompt}

INSTRUCTIONS:
- Analyze the user request and dataset characteristics
- Choose the most appropriate visualization type based on the data and request
- Use exact column names from the schema above
- Respect column roles (temporal, categorical, measure, etc.)
- Include helpful tooltips and labels
- Return ONLY valid JSON matching the schema defined in the system prompt
- Do NOT include code fences, explanations, or any text outside the JSON object
"""

def _validate_ops(ops: list[dict]) -> list[dict]:
    valid: list[dict] = []
    for op in ops or []:
        kind = (op.get("op") or "").strip().lower()
        if kind == "scatter_data":
            x = op.get("x"); y = op.get("y")
            if not isinstance(x, str) or not isinstance(y, str) or not x or not y:
                # skip invalid scatter requests produced by placeholders
                continue
        # add other minimal checks as needed
        valid.append(op)
    return valid


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
_re_single_quoted   = re.compile(r"'([^'\\]*(?:\\.[^'\\]*)*)'")
_re_bare_literals   = re.compile(r"\b(?:None|True|False)\b", re.I)
_re_angle_tokens    = re.compile(r"<[^>]*>")     # e.g., <str>, <bool>, <field name>
_re_ellipses        = re.compile(r"\.\.\.")      # literal ...

def _try_repair_json(s: str) -> dict:
    t = s
    # 1) remove trailing commas
    t = _re_trailing_commas.sub(r"\1", t)
    # 2) Python literals -> JSON
    t = _re_bare_literals.sub(lambda m: {"none":"null","true":"true","false":"false"}[m.group(0).lower()], t)
    # 3) angle-bracket placeholders -> null
    t = _re_angle_tokens.sub("null", t)
    # 4) ellipses -> null
    t = _re_ellipses.sub("null", t)
    # 5) single-quoted strings -> double-quoted
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


def chat_plan_structured(system_prompt: str, user_message: str, use_strict_mode: bool = False) -> Plan:
    """
    Request a structured Plan output from the LLM using Pydantic models.

    Args:
        system_prompt: System prompt defining the task
        user_message: User message with data context
        use_strict_mode: If True, use strict schema validation (OpenAI only)

    Returns:
        Validated Plan object

    Raises:
        LLMError: If structured output fails or validation fails
    """
    llm = _get_llm()
    provider = get_provider_name()

    try:
        # Configure structured output based on provider
        if use_strict_mode and provider in {"openai", "oa"}:
            # OpenAI supports strict mode for better validation
            llm_struct = llm.with_structured_output(Plan, method="json_schema", strict=True)
        else:
            # Standard structured output via tool calling (works for Groq, OpenAI)
            llm_struct = llm.with_structured_output(Plan)

        msg = [SystemMessage(system_prompt), HumanMessage(user_message)]
        logger.debug(f"Requesting structured output from {provider} with strict={use_strict_mode}")

        out = llm_struct.invoke(msg)

        if out is None:
            raise LLMError("structured_output_failed: empty result")

        if isinstance(out, Plan):
            logger.debug("Received valid Plan object from LLM")
            return out

        if isinstance(out, dict):
            logger.debug("Received dict, converting to Plan")
            return Plan(**out)

        # Fallback: parse text response
        logger.warning("Structured output returned unexpected type, attempting text parsing")
        text = _as_text_from_response(out) or _as_text_from_content(
            getattr(out, "content", None)
        )
        block = _extract_json_block(text)
        obj = json.loads(block)
        obj = _coerce_plan_object(obj)
        return Plan(**obj)

    except Exception as e:
        logger.error(f"Structured output failed: {e}")
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


def _short_error(exc: Exception) -> str:
    msg = str(exc)
    if not msg:
        return exc.__class__.__name__
    msg = msg.replace("\n", " ").strip()
    return msg[:200]


def _plan_to_dict(plan: Plan | dict) -> dict:
    if isinstance(plan, Plan):
        return plan.model_dump(by_alias=True, exclude_none=True)
    return plan


def plan_from_llm(
    columns_md: str,
    user_prompt: str,
    client_ctx: dict | None,
    *,
    profile: dict | None = None,
    max_attempts: int = 2,
) -> dict:
    """
    Ask the LLM for a visualization plan with intelligent retry logic.

    Strategy:
    1. Try Pydantic structured output first (if provider supports it)
    2. Fall back to JSON mode with response_format
    3. Last resort: text parsing with repair

    Args:
        columns_md: Markdown representation of columns
        user_prompt: User's natural language query
        client_ctx: Client context (existing visualizations, selections)
        profile: Rich dataset profile with visualization hints
        max_attempts: Maximum retry attempts

    Returns:
        Dictionary representation of the Plan

    Raises:
        LLMError: If all attempts fail
    """

    provider = get_provider_name()
    use_pydantic_first = supports_native_structured_output()

    logger.info(f"Planning with provider={provider}, pydantic_first={use_pydantic_first}")

    attempts: list[dict] = []
    # first attempt uses full profile (if provided); later attempts drop it to reduce prompt size
    if profile is not None:
        attempts.append({"profile": profile, "method": "pydantic" if use_pydantic_first else "json"})

    # Subsequent attempts try different methods
    attempts.append({"profile": None, "method": "pydantic" if use_pydantic_first else "json"})
    attempts.append({"profile": None, "method": "json" if use_pydantic_first else "pydantic"})

    last_error: Exception | None = None

    for i, attempt in enumerate(attempts[:max_attempts]):
        prof = attempt.get("profile")
        method = attempt.get("method", "pydantic")

        # Build user message with context
        user_msg = build_user_prompt(columns_md, user_prompt, client_ctx, profile=prof)

        if last_error is not None:
            user_msg += (
                "\n\nPrevious response could not be parsed because: "
                + _short_error(last_error)
                + "\nReturn EXACTLY one valid JSON object as specified."
            )

        logger.debug(f"Attempt {i+1}/{max_attempts} using method={method}")

        try:
            if method == "pydantic" and use_pydantic_first:
                # Pydantic-first approach (best for OpenAI, Groq)
                plan = chat_plan_structured(SYSTEM_PROMPT, user_msg, use_strict_mode=(provider in {"openai", "oa"}))
                return _plan_to_dict(plan)
            else:
                # JSON mode approach (best for NVIDIA and fallback)
                result = chat_json(SYSTEM_PROMPT, user_msg)
                return result

        except Exception as exc:
            logger.warning(f"Attempt {i+1} failed with {method}: {_short_error(exc)}")
            last_error = exc

            # Try alternate method as immediate fallback
            try:
                if method == "pydantic":
                    logger.debug("Pydantic failed, trying JSON mode")
                    result = chat_json(SYSTEM_PROMPT, user_msg)
                    return result
                else:
                    logger.debug("JSON mode failed, trying Pydantic")
                    plan = chat_plan_structured(SYSTEM_PROMPT, user_msg)
                    return _plan_to_dict(plan)
            except Exception as exc2:
                logger.warning(f"Fallback also failed: {_short_error(exc2)}")
                last_error = exc2
                continue

    # All attempts failed
    if last_error is not None:
        raise LLMError(f"plan_failed_after_{max_attempts}_retries: {_short_error(last_error)}")
    raise LLMError("plan_failed_after_retries: unknown error")

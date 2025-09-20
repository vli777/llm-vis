import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, List
from .llm import plan_from_llm as lc_plan_from_llm, LLMError 

# ---------- helpers ----------

def pick_table(tables: Dict[str, pd.DataFrame]) -> Tuple[str, pd.DataFrame]:
    if not tables:
        raise ValueError("No tables uploaded or preloaded.")
    name = max(tables, key=lambda k: len(tables[k]))
    return name, tables[name]

def schema_hint(df: pd.DataFrame) -> str:
    return "\n".join(f"- {c} ({str(df[c].dtype)})" for c in df.columns)

# ---------- LLM plan ----------

def plan_from_llm(prompt: str, df: pd.DataFrame) -> Dict[str, Any]:
    """
    Ask the LLM for a viz plan (intent/title/operations/vega_lite) given the dataset schema.
    Uses LangChain client with structured output if available, otherwise JSON fallback.
    """
    return lc_plan_from_llm(schema_hint(df), prompt)

# ---------- executor ----------

def _ensure_cols(df: pd.DataFrame, cols: List[str]) -> List[str]:
    present = [c for c in cols if c in df.columns]
    if not present:
        raise ValueError(f"None of the requested columns are in the dataset: {cols}")
    return present

def exec_operations(df: pd.DataFrame, ops: List[Dict[str, Any]]) -> pd.DataFrame:
    result = df.copy()
    for op in (ops or []):
        kind = op.get("op")

        if kind == "value_counts":
            col = op["col"]
            if col not in result.columns:
                raise ValueError(f"Column not found for value_counts: {col}")
            key, ncol = op.get("as", [col, "n"])
            series = result[col].astype("string").fillna("∅")
            out = series.value_counts(dropna=False).reset_index()
            out.columns = [key, ncol]
            result = out

        elif kind == "explode_counts":
            col = op["col"]
            if col not in result.columns:
                raise ValueError(f"Column not found for explode_counts: {col}")
            # default to common delimiters if not provided
            sep = op.get("sep") or r"[;,]"
            key, ncol = op.get("as", ["value", "n"])
            s = result[col].fillna("").astype(str).str.split(sep)
            exploded = result.assign(_value=s).explode("_value")
            cleaned = exploded["_value"].astype(str).str.strip()
            out = (
                cleaned[cleaned.ne("")]
                .str.replace(r"\s+", " ", regex=True)
                .value_counts()
                .reset_index()
            )
            out.columns = [key, ncol]
            result = out

        elif kind == "scatter_data":
            x, y = op["x"], op["y"]
            extras = op.get("extras", [])
            cols = _ensure_cols(result, [x, y] + extras)
            result = result[cols].replace([np.inf, -np.inf], np.nan).dropna()

        elif kind == "corr_pair":
            x, y = op["x"], op["y"]
            if x not in result.columns or y not in result.columns:
                raise ValueError(f"Columns not found for corr_pair: {x}, {y}")
            sub = result[[x, y]].replace([np.inf, -np.inf], np.nan).dropna()
            result.attrs["pearson"] = float(sub[x].corr(sub[y], method="pearson"))
            result.attrs["spearman"] = float(sub[x].corr(sub[y], method="spearman"))

        else:
            raise ValueError(f"Unsupported op: {kind}")
    return result

# ---------- spec attachment ----------

def attach_values_to_spec(spec: Dict[str, Any], df: pd.DataFrame) -> Dict[str, Any]:
    spec = dict(spec or {})
    spec["$schema"] = spec.get("$schema") or "https://vega.github.io/schema/vega-lite/v5.json"
    # We inline values so the frontend doesn't need named datasets
    spec["data"] = {"values": df.to_dict(orient="records")}
    return spec

# ---------- main entry ----------

def handle_llm_nlq(prompt: str, tables: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
    """
    LLM-only path: build a plan from the LLM and execute it via pandas.
    Returns either a table payload or a Vega-Lite spec with inline data.
    """
    name, df = pick_table(tables)
    plan = plan_from_llm(prompt, df)

    intent = plan.get("intent")
    title = plan.get("title") or "Result"
    ops = plan.get("operations", [])
    spec = plan.get("vega_lite")

    data = exec_operations(df, ops)

    if intent == "table" or spec is None:
        return {
            "type": "table",
            "title": title,
            "table": {"columns": list(data.columns), "rows": data.to_dict(orient="records")},
        }

    return {"type": "chart", "title": title, "spec": attach_values_to_spec(spec, data)}

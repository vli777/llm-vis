import re
from typing import Dict, Any, Tuple, List, Optional

import numpy as np
import pandas as pd

from .llm import plan_from_llm as lc_plan_from_llm, columns_markdown  


# ---------- helpers ----------

def pick_table(tables: Dict[str, pd.DataFrame]) -> Tuple[str, pd.DataFrame]:
    """Choose a table to operate on. Currently: largest by row-count."""
    if not tables:
        raise ValueError("No tables uploaded.")
    name = max(tables, key=lambda k: len(tables[k]))
    return name, tables[name]


def schema_hint(df: pd.DataFrame) -> str:
    return "\n".join(f"- {c} ({str(df[c].dtype)})" for c in df.columns)


def resolve_col(name: Optional[str], df: pd.DataFrame) -> Optional[str]:
    """Resolve a column name case-insensitively; tolerate simple spacing/underscore differences."""
    if not name:
        return None
    candidates = list(df.columns)
    lower_map = {c.lower(): c for c in candidates}
    # direct lower match
    key = name.lower()
    if key in lower_map:
        return lower_map[key]
    # normalize spaces/underscores/hyphens
    def norm(s: str) -> str:
        return re.sub(r"[\s_\-]+", "", s.lower())
    target = norm(name)
    for c in candidates:
        if norm(c) == target:
            return c
    # no luck
    return name if name in df.columns else None

def _pct(n: int, d: int) -> float:
    return 0.0 if d <= 0 else round(100.0 * n / d, 2)

def _example_values(s: pd.Series, k: int = 3):
    vals = s.dropna().unique()[:k]
    # stringify but keep short
    return [str(v)[:80] for v in vals]

def _topk_counts(s: pd.Series, k: int = 5) -> list[dict]:
    vc = s.astype("string").fillna("∅").value_counts().head(k)
    return [{"value": str(i)[:80], "n": int(v)} for i, v in vc.items()]

def dataset_profile(
    df: pd.DataFrame,
    *,
    max_cols: int = 30,
    include_quants: bool = True,
) -> Dict[str, Any]:
    """
    Compact, LLM-friendly profile of the current table.
    Keep it small: truncate long strings and cap lists.
    """
    nrows = len(df)
    cols = []
    for i, c in enumerate(df.columns):
        if i >= max_cols: 
            break
        s = df[c]
        dtype = str(s.dtype)
        missing = int(s.isna().sum())
        unique = int(s.nunique(dropna=True))
        info: Dict[str, Any] = {
            "name": c,
            "dtype": dtype,
            "missing_pct": _pct(missing, nrows),
            "unique": unique,
            "examples": _example_values(s, 3),
        }

        if pd.api.types.is_numeric_dtype(s):
            s_num = pd.to_numeric(s, errors="coerce")
            info["num_stats"] = {
                "min": float(s_num.min()) if nrows else None,
                "max": float(s_num.max()) if nrows else None,
                "mean": float(s_num.mean()) if nrows else None,
                "std": float(s_num.std()) if nrows else None,
            }
            if include_quants:
                qs = s_num.quantile([0.25, 0.5, 0.75]).to_dict()
                info["num_stats"].update({f"q{int(q*100)}": float(v) for q, v in qs.items() if not math.isnan(v)})
        elif pd.api.types.is_datetime64_any_dtype(s):
            s_dt = pd.to_datetime(s, errors="coerce")
            info["datetime_range"] = {
                "min": s_dt.min().isoformat() if s_dt.notna().any() else None,
                "max": s_dt.max().isoformat() if s_dt.notna().any() else None,
            }
        else:
            info["top_values"] = _topk_counts(s, 5)

        cols.append(info)

    profile: Dict[str, Any] = {"row_count": nrows, "columns": cols}

    # 👇 add 1–3 sample rows (stringified and truncated) for extra context
    if nrows > 0:
        sample = (
            df.sample(min(3, nrows), random_state=0)
            .astype(str)
            .applymap(lambda s: s[:80])
            .to_dict(orient="records")
        )
        profile["sample_rows"] = sample

    return profile

# ---------- LLM plan ----------

def plan_from_llm(prompt: str, df: pd.DataFrame, client_ctx: Optional[dict]) -> Dict[str, Any]:
    prof = dataset_profile(df, max_cols=30)
    cols_md = columns_markdown(df)
    # pass both schema (markdown) and profile (JSON) down
    return lc_plan_from_llm(cols_md, prompt, client_ctx or {}, profile=prof)


# ---------- execution primitives ----------

def _ensure_cols(df: pd.DataFrame, cols: List[str]) -> List[str]:
    resolved = []
    for c in cols:
        rc = resolve_col(c, df)
        if rc is None:
            continue
        resolved.append(rc)
    present = [c for c in resolved if c in df.columns]
    if not present:
        raise ValueError(f"None of the requested columns are in the dataset: {cols}")
    return present


def _coerce_numeric_inplace(df: pd.DataFrame, cols: List[str]) -> None:
    """Convert currency/commas/percent strings to numeric where possible."""
    for c in set(cols):
        if c not in df.columns:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            continue
        # strip common junk then to_numeric
        df[c] = (
            df[c]
            .astype("string")
            .str.replace(r"[,\$\u00A0]", "", regex=True)
            .str.replace(r"%$", "", regex=True)
        )
        df[c] = pd.to_numeric(df[c], errors="coerce")


def exec_operations(df: pd.DataFrame, ops: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Execute a small DSL of operations safely in pandas.
    Supported ops: value_counts, explode_counts, scatter_data, corr_pair
    """
    result = df.copy()

    for op in (ops or []):
        kind = op.get("op")

        if kind == "value_counts":
            col = resolve_col(op.get("col"), result)
            if not col or col not in result.columns:
                raise ValueError(f"Column not found for value_counts: {op.get('col')}")
            key, ncol = (op.get("as") or [col, "n"])
            series = result[col].astype("string").fillna("∅")
            out = series.value_counts(dropna=False).reset_index()
            out.columns = [key, ncol]
            result = out

        elif kind == "explode_counts":
            # If LLM forgot the column, heuristically try investor-like columns
            col = resolve_col(op.get("col"), result)
            if not col or col not in result.columns:
                # heuristic fallback for common multi-value fields
                for guess in ["investors", "backers", "lead_investors", "tags"]:
                    g = resolve_col(guess, result)
                    if g and g in result.columns:
                        col = g
                        break
            if not col or col not in result.columns:
                raise ValueError(f"Column not found for explode_counts: {op.get('col')}")

            sep = op.get("sep") or r"[;,/|]"
            key, ncol = (op.get("as") or ["value", "n"])

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
            x = resolve_col(op.get("x"), result)
            y = resolve_col(op.get("y"), result)
            extras = [resolve_col(e, result) for e in (op.get("extras") or [])]
            cols = _ensure_cols(result, [x, y] + [e for e in extras if e])
            _coerce_numeric_inplace(result, [x, y])
            result = result[cols].replace([np.inf, -np.inf], np.nan).dropna()

        elif kind == "corr_pair":
            x = resolve_col(op.get("x"), result)
            y = resolve_col(op.get("y"), result)
            if not x or not y or x not in result.columns or y not in result.columns:
                raise ValueError(f"Columns not found for corr_pair: {op.get('x')}, {op.get('y')}")
            _coerce_numeric_inplace(result, [x, y])
            sub = result[[x, y]].replace([np.inf, -np.inf], np.nan).dropna()
            if len(sub) == 0:
                result.attrs["pearson"] = float("nan")
                result.attrs["spearman"] = float("nan")
            else:
                result.attrs["pearson"] = float(sub[x].corr(sub[y], method="pearson"))
                result.attrs["spearman"] = float(sub[x].corr(sub[y], method="spearman"))

        else:
            raise ValueError(f"Unsupported op: {kind}")

    # Safety: cap huge payloads sent back to the FE for speed
    if len(result) > 5000:
        result = result.head(5000).copy()
    return result


# ---------- spec attachment ----------

def attach_values_to_spec(spec: Dict[str, Any], df: pd.DataFrame) -> Dict[str, Any]:
    spec = dict(spec or {})
    spec["$schema"] = spec.get("$schema") or "https://vega.github.io/schema/vega-lite/v5.json"
    spec["data"] = {"values": df.to_dict(orient="records")}
    return spec


# ---------- main entry ----------

def handle_llm_nlq(prompt: str, tables: Dict[str, pd.DataFrame], client_ctx: Optional[dict] = None) -> Dict[str, Any]:
    """
    LLM-only path: the model returns either:
      - action="create" -> we compute ops and return a chart/table
      - action="update" -> we pass JSON Patch + target info to the FE to patch an existing chart
    """
    _, df = pick_table(tables)
    plan = plan_from_llm(prompt, df, client_ctx)

    # Normalize keys the model might vary on
    action = (plan.get("action") or "create").lower()
    vtype = (plan.get("type") or plan.get("intent") or "chart").lower()

    if action == "update":
        patch = plan.get("patch") or []
        if not isinstance(patch, list) or not patch:
            raise ValueError("Update requested but no patch provided.")
        # Targeting info (optional): targetId or 'last'
        payload = {"type": vtype, "action": "update", "patch": patch}
        if plan.get("targetId"):
            payload["targetId"] = plan["targetId"]
        if plan.get("target"):
            payload["target"] = plan["target"]
        return payload

    # action == "create"
    title = plan.get("title") or "Result"
    ops = plan.get("operations", [])
    spec = plan.get("vega_lite")

    data = exec_operations(df, ops)

    if vtype == "table" or spec is None:
        return {
            "type": "table",
            "action": "create",
            "title": title,
            "table": {"columns": list(data.columns), "rows": data.to_dict(orient="records")},
        }

    return {"type": "chart", "action": "create", "title": title, "spec": attach_values_to_spec(spec, data)}

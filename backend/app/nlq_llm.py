import re
import copy
from typing import Dict, Any, Tuple, List, Optional

import numpy as np
import pandas as pd
import math
import logging
from datetime import datetime

from .llm import (
    plan_from_llm as lc_plan_from_llm,
    columns_markdown,
    _coerce_plan_object,
    _validate_ops,
)


logger = logging.getLogger("uvicorn.error")


# ---------- helpers ----------
def df_json_safe(df: pd.DataFrame) -> pd.DataFrame:
    """Replace ±inf -> NaN, then NaN -> None so Starlette JSONResponse can serialize."""
    if df.empty:
        return df
    tmp = df.replace([np.inf, -np.inf], np.nan)
    # ensure we can hold None
    tmp = tmp.astype(object)
    return tmp.where(pd.notna(tmp), None)


def df_to_records_safe(df: pd.DataFrame) -> list[dict]:
    return df_json_safe(df).to_dict(orient="records")


def _parse_created_at(meta: dict) -> float:
    created = meta.get("created_at") if isinstance(meta, dict) else None
    if isinstance(created, str) and created:
        try:
            return datetime.fromisoformat(created.replace("Z", "")).timestamp()
        except ValueError:
            return 0.0
    return 0.0


def pick_table(
    tables: Dict[str, pd.DataFrame],
    meta_store: Optional[Dict[str, dict]] = None,
) -> Tuple[str, pd.DataFrame]:
    """Choose the most recent table based on metadata (falls back to insertion order)."""
    if not tables:
        raise ValueError("No tables uploaded.")

    if meta_store:
        def sort_key(name: str) -> Tuple[float, int]:
            meta = meta_store.get(name) or {}
            ts = _parse_created_at(meta)
            # preserve insertion order as secondary key
            try:
                order_idx = list(tables.keys()).index(name)
            except ValueError:
                order_idx = -1
            return (ts, order_idx)

        latest_name = max(tables.keys(), key=sort_key)
        return latest_name, tables[latest_name]

    try:
        latest_name = next(reversed(tables.keys()))
    except StopIteration:  # pragma: no cover - defensive
        raise ValueError("No tables uploaded.")
    return latest_name, tables[latest_name]


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


def _fix_field(value, df):
    if not isinstance(value, str):
        return value
    resolved = resolve_col(value, df)
    return resolved or value


def _canon_fields_in_spec(spec: dict, df: pd.DataFrame) -> dict:
    s = copy.deepcopy(spec)

    def walk(x):
        if isinstance(x, dict):
            # common Vega-Lite places
            if "field" in x:
                x["field"] = _fix_field(x["field"], df)
            if "aggregate" in x and isinstance(x["aggregate"], dict):
                agg_obj = x["aggregate"]
                op = None
                if isinstance(agg_obj, dict):
                    op = (
                        agg_obj.get("op")
                        or agg_obj.get("aggregate")
                        or agg_obj.get("name")
                    )
                if op:
                    x["aggregate"] = op
                else:
                    x.pop("aggregate", None)
            if "groupby" in x and isinstance(x["groupby"], list):
                x["groupby"] = [_fix_field(v, df) for v in x["groupby"]]
            if "fields" in x and isinstance(x["fields"], list):
                x["fields"] = [_fix_field(v, df) for v in x["fields"]]
            if "sort" in x and isinstance(x["sort"], dict) and "field" in x["sort"]:
                x["sort"]["field"] = _fix_field(x["sort"]["field"], df)
            if "aggregate" in x and isinstance(x["aggregate"], str):
                # scrub stray aggregate on top-level encode objects like transform aggregate
                if x["aggregate"].strip() == "":
                    x.pop("aggregate", None)
            if "transform" in x and isinstance(x["transform"], list):
                x["transform"] = [walk(v) for v in x["transform"]]
            for k, v in x.items():
                x[k] = walk(v)
        elif isinstance(x, list):
            return [walk(v) for v in x]
        return x

    return walk(s)


def _normalize_pie_chart(spec: dict, data: pd.DataFrame) -> dict:
    mark = spec.get("mark")

    def _mark_type(m) -> str:
        if isinstance(m, str):
            return m.lower()
        if isinstance(m, dict):
            return (m.get("type") or "").lower()
        return ""

    mtype = _mark_type(mark)
    if mtype not in {"pie", "donut", "doughnut", "arc"}:
        return spec

    # Normalize mark -> arc with filled default
    if isinstance(mark, dict):
        normalized_mark = {**mark, "type": "arc", "filled": True}
    else:
        normalized_mark = {"type": "arc", "filled": True}
    spec["mark"] = normalized_mark

    enc = spec.setdefault("encoding", {})

    # Derive category field from color/x
    category_field = None
    if isinstance(enc.get("color"), dict):
        category_field = enc["color"].get("field")
    if not category_field and isinstance(enc.get("x"), dict):
        category_field = enc["x"].get("field")

    if category_field and "color" not in enc:
        enc["color"] = {"field": category_field, "type": enc.get("x", {}).get("type", "nominal")}

    # Ensure theta channel exists
    theta = enc.get("theta") if isinstance(enc.get("theta"), dict) else None

    if not theta or not theta.get("field"):
        numeric_cols = [c for c in data.columns if pd.api.types.is_numeric_dtype(data[c])]
        preferred = None
        for name in [
            "count",
            "n",
            "value",
            "total",
            "sum",
        ]:
            if name in data.columns and name in numeric_cols:
                preferred = name
                break
        if not preferred and numeric_cols:
            preferred = numeric_cols[0]

        if preferred:
            enc["theta"] = {"field": preferred, "type": "quantitative"}
        else:
            enc["theta"] = {"aggregate": "count"}

    # Preserve category info in tooltip when removing x
    if category_field:
        tooltips = enc.setdefault("tooltip", [])
        if isinstance(tooltips, dict):
            tooltips = [tooltips]
        if isinstance(tooltips, list):
            existing = {t.get("field") for t in tooltips if isinstance(t, dict) and t.get("field")}
            if category_field not in existing:
                tooltips.append({"field": category_field})
            enc["tooltip"] = tooltips

    enc.pop("x", None)
    enc.pop("y", None)

    # Drop transforms that only attempted aggregate counts (theta handles it)
    transforms = spec.get("transform")
    if isinstance(transforms, list):
        cleaned: list[dict] = []
        for t in transforms:
            if not isinstance(t, dict):
                continue
            if not t:
                continue
            agg = t.get("aggregate")
            if isinstance(agg, str) and agg.strip().lower() in {"count", "sum", "mean"}:
                continue
            cleaned.append(t)
        if cleaned:
            spec["transform"] = cleaned
        else:
            spec.pop("transform", None)

    return spec


PROFILE_MAX_ROWS = 2000


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
    work_df = df
    if nrows > PROFILE_MAX_ROWS:
        # sample deterministically so repeated calls stay stable and fast
        work_df = df.sample(PROFILE_MAX_ROWS, random_state=0)
    work_rows = len(work_df)
    cols = []
    for i, c in enumerate(df.columns):
        if i >= max_cols:
            break
        s = df[c]
        s_sample = work_df[c] if c in work_df.columns else s
        dtype = str(s.dtype)
        missing = int(s.isna().sum())
        unique = int(s_sample.nunique(dropna=True))
        info: Dict[str, Any] = {
            "name": c,
            "dtype": dtype,
            "missing_pct": _pct(missing, nrows),
            "unique": unique,
            "examples": _example_values(s_sample, 3),
        }

        if pd.api.types.is_numeric_dtype(s):
            s_num = pd.to_numeric(s_sample, errors="coerce")
            info["num_stats"] = {
                "min": float(s_num.min()) if work_rows else None,
                "max": float(s_num.max()) if work_rows else None,
                "mean": float(s_num.mean()) if work_rows else None,
                "std": float(s_num.std()) if work_rows else None,
            }
            if include_quants:
                qs = s_num.quantile([0.25, 0.5, 0.75]).to_dict()
                info["num_stats"].update(
                    {
                        f"q{int(q * 100)}": float(v)
                        for q, v in qs.items()
                        if not math.isnan(v)
                    }
                )
        elif pd.api.types.is_datetime64_any_dtype(s):
            s_dt = pd.to_datetime(s_sample, errors="coerce")
            info["datetime_range"] = {
                "min": s_dt.min().isoformat() if s_dt.notna().any() else None,
                "max": s_dt.max().isoformat() if s_dt.notna().any() else None,
            }
        else:
            info["top_values"] = _topk_counts(s_sample, 5)

        cols.append(info)

    profile: Dict[str, Any] = {"row_count": nrows, "columns": cols}

    # 👇 add 1–3 sample rows (stringified and truncated) for extra context
    if work_rows > 0:
        sample = (
            work_df.sample(min(3, work_rows), random_state=0)
            .astype(str)
            .map(lambda s: s[:80])
            .to_dict(orient="records")
        )
        profile["sample_rows"] = sample

    return profile


# ---------- LLM plan ----------


def plan_from_llm(
    prompt: str, df: pd.DataFrame, client_ctx: Optional[dict]
) -> Dict[str, Any]:
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

    Additions:
    - value_counts/explode_counts: optional `limit` (int) to cap top-N.
    - scatter_data: optional `log` (bool) to log-transform x and y; drops <= 0 before log.
    - corr_pair: stores JSON-safe floats in result.attrs (None when not finite).
    """
    result = df.copy()

    for op in ops or []:
        kind = op.get("op")

        if kind == "value_counts":
            col = resolve_col(op.get("col"), result)
            if not col or col not in result.columns:
                raise ValueError(f"Column not found for value_counts: {op.get('col')}")
            key, ncol = op.get("as") or [col, "n"]
            limit = op.get("limit")
            series = result[col].astype("string").fillna("∅")

            sep = op.get("sep")
            if not sep:
                sample = series.head(100)
                if sample.str.contains(r"[;,/|]").mean() > 0.2:
                    sep = r"[;,/|]"

            if sep:
                splitted = (
                    series.fillna("")
                    .astype(str)
                    .str.split(sep)
                    .explode()
                    .astype(str)
                    .str.strip()
                    .str.replace(r"\s+", " ", regex=True)
                )
                series = splitted[splitted.ne("")]

            out = series.value_counts(dropna=False)
            if isinstance(limit, int) and limit > 0:
                out = out.head(limit)
            out = out.reset_index()
            out.columns = [key, ncol]
            result = out

        elif kind == "explode_counts":
            col = resolve_col(op.get("col"), result)
            if not col or col not in result.columns:
                # heuristic fallback for common multi-value fields
                for guess in ["investors", "backers", "lead_investors", "tags"]:
                    g = resolve_col(guess, result)
                    if g and g in result.columns:
                        col = g
                        break
            if not col or col not in result.columns:
                raise ValueError(
                    f"Column not found for explode_counts: {op.get('col')}"
                )

            sep = op.get("sep") or r"[;,/|]"
            key, ncol = op.get("as") or ["value", "n"]
            limit = op.get("limit")

            s = result[col].fillna("").astype(str).str.split(sep)
            exploded = result.assign(_value=s).explode("_value")
            cleaned = exploded["_value"].astype(str).str.strip()
            counts = (
                cleaned[cleaned.ne("")]
                .str.replace(r"\s+", " ", regex=True)
                .value_counts()
            )
            if isinstance(limit, int) and limit > 0:
                counts = counts.head(limit)
            out = counts.reset_index()
            out.columns = [key, ncol]
            result = out

        elif kind == "scatter_data":
            x = resolve_col(op.get("x"), result)
            y = resolve_col(op.get("y"), result)
            extras = [resolve_col(e, result) for e in (op.get("extras") or [])]
            cols = _ensure_cols(result, [x, y] + [e for e in extras if e])

            # numeric coercion for x,y; extras are passed through
            _coerce_numeric_inplace(result, [x, y])

            # Optional log transform
            if op.get("log"):
                # drop non-positive before log
                result = result[
                    (pd.to_numeric(result[x], errors="coerce") > 0)
                    & (pd.to_numeric(result[y], errors="coerce") > 0)
                ]
                result[x] = np.log(pd.to_numeric(result[x], errors="coerce"))
                result[y] = np.log(pd.to_numeric(result[y], errors="coerce"))

            result = (
                result[cols].replace([np.inf, -np.inf], np.nan).dropna(subset=[x, y])
            )

        elif kind == "corr_pair":
            x = resolve_col(op.get("x"), result)
            y = resolve_col(op.get("y"), result)
            if not x or not y or x not in result.columns or y not in result.columns:
                raise ValueError(
                    f"Columns not found for corr_pair: {op.get('x')}, {op.get('y')}"
                )
            _coerce_numeric_inplace(result, [x, y])
            sub = result[[x, y]].replace([np.inf, -np.inf], np.nan).dropna()
            if len(sub) == 0:
                result.attrs["pearson"] = None
                result.attrs["spearman"] = None
            else:
                p = sub[x].corr(sub[y], method="pearson")
                s = sub[x].corr(sub[y], method="spearman")
                result.attrs["pearson"] = (
                    float(p) if (p is not None and math.isfinite(p)) else None
                )
                result.attrs["spearman"] = (
                    float(s) if (s is not None and math.isfinite(s)) else None
                )

        else:
            raise ValueError(f"Unsupported op: {kind}")

    # Safety: cap huge payloads sent back to the FE for speed
    if len(result) > 5000:
        result = result.head(5000).copy()

    return result


# ---------- spec attachment ----------


def _sanitize_for_spec(df: pd.DataFrame, spec: Dict[str, Any]) -> pd.DataFrame:
    if df.empty or not isinstance(spec, dict):
        return df

    enc = spec.get("encoding")
    if not isinstance(enc, dict):
        return df

    out = df.copy()
    numeric_fields: List[str] = []

    for channel in ("x", "x2", "y", "y2", "theta", "size"):
        node = enc.get(channel)
        if not isinstance(node, dict):
            continue
        field = node.get("field")
        if not isinstance(field, str) or field not in out.columns:
            continue
        kind = (node.get("type") or "").lower()
        if kind == "quantitative":
            out[field] = pd.to_numeric(out[field], errors="coerce")
            numeric_fields.append(field)

    if numeric_fields:
        out = out.replace([np.inf, -np.inf], np.nan)
        out = out.dropna(subset=numeric_fields)
        if out.empty:
            return df

    return out


def attach_values_to_spec(spec: Dict[str, Any], df: pd.DataFrame) -> Dict[str, Any]:
    spec = dict(spec or {})
    spec["$schema"] = (
        spec.get("$schema") or "https://vega.github.io/schema/vega-lite/v5.json"
    )
    spec["data"] = {"values": df_to_records_safe(df)}  # <-- SAFE
    return spec


# ---------- main entry ----------

import json


def handle_llm_nlq(
    prompt: str,
    tables: Dict[str, pd.DataFrame],
    client_ctx: Optional[dict] = None,
    *,
    meta_store: Optional[Dict[str, dict]] = None,
) -> Dict[str, Any]:
    # pick a table
    if meta_store:
        # keep meta store in sync with live tables to avoid stale entries
        stale = [name for name in meta_store.keys() if name not in tables]
        for name in stale:
            meta_store.pop(name, None)

    table_name, df = pick_table(tables, meta_store)

    # plan (coerce list/str -> dict)
    plan = plan_from_llm(prompt, df, client_ctx)
    plan = _coerce_plan_object(plan)

    # normalize keys
    action = (plan.get("action") or "create").lower()
    vtype = (plan.get("type") or plan.get("intent") or "chart").lower()

    # ----- UPDATE path -----
    if action == "update":
        patch = plan.get("patch") or []
        if not isinstance(patch, list) or not patch:
            raise ValueError("Update requested but no patch provided.")
        payload = {"type": vtype, "action": "update", "patch": patch}
        if plan.get("targetId"):
            payload["targetId"] = plan["targetId"]
        if plan.get("target"):
            payload["target"] = plan["target"]
        return payload

    # ----- CREATE path -----
    title = plan.get("title") or "Result"

    # normalize + validate operations
    ops_raw = plan.get("operations") or []
    if not isinstance(ops_raw, list):
        ops_raw = [ops_raw] if isinstance(ops_raw, dict) else []
    ops = _validate_ops(ops_raw)

    spec = plan.get("vega_lite")

    # run backend ops to produce the data used by either table or chart
    data = exec_operations(df, ops)

    # decide on table vs chart
    spec = plan.get("vega_lite")

    # TABLE (or no spec)
    if vtype == "table" or spec is None:
        return {
            "type": "table",
            "action": "create",
            "title": title,
            "table": {"columns": list(data.columns), "rows": df_to_records_safe(data)},
        }

    # CHART
    # tolerate the spec being a JSON string
    if isinstance(spec, str):
        try:
            spec = json.loads(spec)
        except Exception:
            raise ValueError("vega_lite must be an object or valid JSON string")

    if not isinstance(spec, dict):
        raise ValueError("vega_lite must be an object")

    # fix field casing/spelling to match dataframe columns
    spec = _canon_fields_in_spec(spec, data)
    spec = _normalize_pie_chart(spec, data)

    if logger.isEnabledFor(logging.INFO):
        try:
            spec_for_log = {k: v for k, v in spec.items() if k != "data"}
            logger.info("VEGA SPEC (from LLM): %s", json.dumps(spec_for_log, indent=2))
        except Exception:
            logger.info("VEGA SPEC (from LLM): %s", spec)

    # the backend injects data; remove any model-provided data block to avoid conflicts
    if "data" in spec:
        spec.pop("data", None)

    clean_data = _sanitize_for_spec(data, spec)

    return {
        "type": "chart",
        "action": "create",
        "title": title,
        "spec": attach_values_to_spec(spec, clean_data),
    }

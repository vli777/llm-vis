import re
import copy
from typing import Dict, Any, Tuple, List, Optional, Iterable

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


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


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


# mapping used for smart numeric parsing
_SUFFIX_MAP = {
    "k": 1_000.0,
    "m": 1_000_000.0,
    "mm": 1_000_000.0,
    "b": 1_000_000_000.0,
    "bn": 1_000_000_000.0,
    "t": 1_000_000_000_000.0,
}


def _nice_bounds(lo: float, hi: float) -> Optional[Tuple[float, float]]:
    if not (math.isfinite(lo) and math.isfinite(hi)):
        return None
    if lo == hi:
        span = abs(lo) if lo else 1.0
        lo -= span * 0.05 + 1.0
        hi += span * 0.05 + 1.0

    span = hi - lo
    if span <= 0:
        span = abs(lo) if lo else 1.0
    buffer = max(span * 0.05, 1.0)
    lo_adj = lo - buffer
    hi_adj = hi + buffer
    rng = hi_adj - lo_adj
    if rng <= 0:
        rng = abs(lo_adj) if lo_adj else 1.0
    exponent = math.floor(math.log10(rng)) if rng > 0 else 0
    step = 10 ** exponent
    nice_lo = math.floor(lo_adj / step) * step
    nice_hi = math.ceil(hi_adj / step) * step
    return nice_lo, nice_hi


def _apply_axis_format(node: dict, field: str, df: pd.DataFrame) -> None:
    """Attach sensible formatting/scales for known numeric fields."""

    if not isinstance(node, dict):
        return

    axis: dict = node.get("axis") if isinstance(node.get("axis"), dict) else {}
    scale: dict = node.get("scale") if isinstance(node.get("scale"), dict) else {}
    node_type = (node.get("type") or "").lower()

    # Examine the data to decide whether zero should be included
    series = None
    if df is not None and field in df.columns:
        if node_type == "temporal":
            series = df[field]
            if pd.api.types.is_datetime64_any_dtype(series):
                valid = series.dropna()
                if not valid.empty:
                    months = valid.dt.month
                    days = valid.dt.day
                    times = valid.dt.normalize()
                    years = valid.dt.year
                    if (
                        (months == 1).all()
                        and (days == 1).all()
                        and (valid == times).all()
                        and (years >= 1000).all()
                        and (years <= 3000).all()
                    ):
                        axis.setdefault("format", "%Y")
                        node.setdefault("timeUnit", "year")
            if isinstance(scale.get("domain"), list) and scale["domain"]:
                if all(isinstance(v, (int, float)) for v in scale["domain"]):
                    scale.pop("domain", None)
            if axis:
                node["axis"] = axis
            if scale:
                node["scale"] = scale
            return
        series = _smart_numeric_series(df[field])
        valid = series.dropna()
        if not valid.empty:
            fmt_hint = _infer_axis_format(df[field])
            if fmt_hint:
                axis.setdefault("format", fmt_hint)
            min_val = float(valid.min())
            max_val = float(valid.max())
            # When all values are strictly positive or strictly negative, avoid forcing zero
            bounds = _nice_bounds(min_val, max_val)
            if min_val > 0 or max_val < 0:
                scale.setdefault("zero", False)
            if bounds and "domain" not in scale:
                lo, hi = bounds
                if lo < min_val:
                    lo = min_val
                if hi > max_val:
                    hi = max_val
                scale["domain"] = [lo, hi]
            if (valid.round() == valid).all():
                if 1000 <= min_val <= 3000 and 1000 <= max_val <= 3000:
                    axis.setdefault("format", "d")
            if "format" not in axis and max_val >= 1_000_000:
                axis.setdefault("format", "~s")

    if axis:
        node["axis"] = axis
    if scale:
        node["scale"] = scale


def _smart_numeric_value(val) -> float:
    if val is None:
        return np.nan
    if isinstance(val, (int, float)):
        return float(val)
    text = str(val).strip()
    if not text:
        return np.nan
    text = text.replace(",", "").replace("$", "").replace("€", "")
    if text.endswith("%"):
        inner = text[:-1].strip()
        try:
            return float(inner) / 100.0
        except ValueError:
            return np.nan
    lower = text.lower()
    if lower in {"n/a", "na", "nan", "none", "null", "-", "--", "—"}:
        return np.nan

    for suffix in sorted(_SUFFIX_MAP.keys(), key=len, reverse=True):
        if lower.endswith(suffix):
            num_part = text[: -len(suffix)]
            try:
                return float(num_part) * _SUFFIX_MAP[suffix]
            except ValueError:
                return np.nan

    try:
        return float(text)
    except ValueError:
        return np.nan


def _smart_numeric_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce")
    converted = series.map(_smart_numeric_value)
    return pd.to_numeric(converted, errors="coerce")


def _infer_axis_format(series: pd.Series) -> Optional[str]:
    """Infer a d3-format string based on raw value patterns."""
    if series.empty:
        return None
    sample = series.dropna()
    if sample.empty:
        return None

    text = sample.astype("string").str.strip()
    if text.empty:
        return None

    n = len(text)
    if n == 0:
        return None

    def ratio(mask: pd.Series) -> float:
        try:
            return float(mask.mean())
        except Exception:
            return 0.0

    dollar = ratio(text.str.contains(r"\\$"))
    euro = ratio(text.str.contains(r"€"))
    pound = ratio(text.str.contains(r"£"))
    if max(dollar, euro, pound) >= 0.2:
        if dollar >= max(euro, pound):
            return "$~s"
        if euro >= pound:
            return "€~s"
        return "£~s"

    if ratio(text.str.contains(r"%$")) >= 0.2:
        return ".0%"

    if ratio(text.str.contains(r"(?i)\\d\\s*[kmbt]\\b")) >= 0.2:
        return "~s"

    return None


def _maybe_coerce_year_temporal(series: pd.Series) -> Optional[pd.Series]:
    """Coerce year-like values into datetime (year precision) for temporal charts."""
    if series.empty:
        return None
    if pd.api.types.is_datetime64_any_dtype(series):
        return series

    if pd.api.types.is_numeric_dtype(series):
        vals = pd.to_numeric(series, errors="coerce")
        valid = vals.dropna()
        if valid.empty:
            return None
        if not (valid.round() == valid).all():
            return None
        if not ((valid >= 1000) & (valid <= 3000)).all():
            return None
        years = pd.to_numeric(series, errors="coerce").round().astype("Int64")
        return pd.to_datetime(years.astype("string"), format="%Y", errors="coerce")

    if pd.api.types.is_string_dtype(series) or pd.api.types.is_object_dtype(series):
        year_str = series.astype("string").str.extract(r"(\d{4})", expand=False)
        dt = pd.to_datetime(year_str, format="%Y", errors="coerce")
        if dt.isna().all():
            return None
        return dt

    return None


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

    cleaned = walk(s)
    if isinstance(cleaned, dict):
        stray_keys = {"x", "y", "size", "opacity", "shape", "color", "theta", "tooltip"}
        keep = {"encoding", "mark", "$schema", "title", "params", "config"}
        for key in list(cleaned.keys()):
            if key in stray_keys and key not in keep:
                cleaned.pop(key, None)
    return cleaned


def _get_mark_type(mark) -> str:
    if isinstance(mark, str):
        return mark.lower()
    if isinstance(mark, dict):
        return (mark.get("type") or "").lower()
    return ""


def _collect_column_stats(df: pd.DataFrame) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    total = len(df)
    numeric_info: List[Dict[str, Any]] = []
    categorical_info: List[Dict[str, Any]] = []

    for col in df.columns:
        tokens = set(_tokenize(col))
        series = df[col]

        # Try to parse as numeric, including string-encoded values like "$1.3B"
        numeric_series = _smart_numeric_series(series)
        non_na = numeric_series.notna().sum()
        parseable_ratio = non_na / total if total > 0 else 0.0

        # Consider it numeric if >50% of values are parseable as numbers
        # This catches both native numeric types AND formatted strings like "$1.3B"
        if total > 0 and non_na >= max(3, int(total * 0.05)) and parseable_ratio > 0.5:
            variance = float(numeric_series.var(skipna=True)) if non_na >= 2 else 0.0
            if not math.isfinite(variance):
                variance = 0.0
            numeric_info.append(
                {
                    "name": col,
                    "variance": variance,
                    "tokens": tokens,
                }
            )
            continue

        unique = series.astype("string").nunique(dropna=True)
        ratio = unique / total if total else 0.0
        categorical_info.append(
            {
                "name": col,
                "unique": unique,
                "ratio": ratio,
                "tokens": tokens,
            }
        )

    return numeric_info, categorical_info


def _choose_numeric_column(
    prompt_tokens: Iterable[str],
    numeric_info: List[Dict[str, Any]],
    *,
    exclude: Iterable[str] = (),
) -> Optional[str]:
    exclude_set = set(exclude)
    tokens = set(prompt_tokens)
    best: Optional[str] = None
    best_score: Tuple[int, int, float] = (-1, -1, -1.0)

    for info in numeric_info:
        name = info["name"]
        if name in exclude_set:
            continue
        overlap = len(tokens & info["tokens"])
        score = (
            1 if overlap > 0 else 0,
            overlap,
            info.get("variance", 0.0),
        )
        if best is None or score > best_score:
            best = name
            best_score = score

    if best is None:
        candidates = [info for info in numeric_info if info["name"] not in exclude_set]
        if candidates:
            best = max(candidates, key=lambda i: i.get("variance", 0.0))["name"]
    return best


def _choose_categorical_column(
    prompt_tokens: Iterable[str],
    categorical_info: List[Dict[str, Any]],
) -> Optional[str]:
    tokens = set(prompt_tokens)
    best: Optional[str] = None
    best_score: Tuple[int, int, float, float, int] = (-1, -1, float("-inf"), float("-inf"), -1)

    for info in categorical_info:
        overlap = len(tokens & info.get("tokens", set()))
        ratio = info.get("ratio", 0.0)
        ratio_score = -abs(ratio - 0.2)
        rarity_score = -ratio  # prefer lower ratio (more repetitions)
        unique = info.get("unique", 0)
        score = (
            1 if overlap > 0 else 0,
            overlap,
            ratio_score,
            rarity_score,
            unique,
        )
        if best is None or score > best_score:
            best = info["name"]
            best_score = score

    if best is None and categorical_info:
        best = max(categorical_info, key=lambda i: i.get("unique", 0))["name"]
    return best


def _auto_operations(
    prompt_tokens: Iterable[str],
    categorical_info: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], bool]:
    tokens = set(prompt_tokens)
    ops: List[Dict[str, Any]] = []
    prefer_table = False

    chart_hints = {
        "chart",
        "graph",
        "plot",
        "visualize",
        "visualise",
        "visualization",
        "visualisation",
        "pie",
        "bar",
        "line",
        "scatter",
        "map",
        "heatmap",
    }

    if tokens & chart_hints:
        return ops, prefer_table

    count_hints = {
        "count",
        "counts",
        "frequency",
        "frequencies",
        "frequent",
        "popular",
        "common",
        "top",
        "most",
        "breakdown",
        "distribution",
        "share",
        "mix",
    }

    if tokens & count_hints:
        col = _choose_categorical_column(tokens, categorical_info)
        if col:
            ops.append(
                {
                    "op": "value_counts",
                    "col": col,
                    "as": [col, "n"],
                    "limit": 25,
                }
            )
            prefer_table = True

    return ops, prefer_table


def _normalize_pie_chart(spec: dict, data: pd.DataFrame) -> dict:
    mark = spec.get("mark")
    mtype = _get_mark_type(mark)
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

    # remove stray top-level encodings that might remain from normalization heuristics
    for channel in ["x", "y", "size", "opacity", "shape", "detail"]:
        if channel in enc and not isinstance(enc[channel], dict):
            enc.pop(channel, None)

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


def _enforce_prompt_preferences(
    spec: dict,
    prompt_tokens: Iterable[str],
    df: pd.DataFrame,
    *,
    numeric_info: List[Dict[str, Any]],
    categorical_info: List[Dict[str, Any]],
) -> dict:
    if not isinstance(spec, dict):
        return spec

    enc = spec.setdefault("encoding", {})
    if not isinstance(enc, dict):
        return spec

    tokens = set(prompt_tokens)
    mark_type = _get_mark_type(spec.get("mark"))

    if mark_type in {"pie", "donut", "doughnut", "arc"}:
        enc.pop("x", None)
        enc.pop("y", None)
        enc.pop("size", None)
        enc.pop("opacity", None)
        enc.pop("shape", None)
        return spec

    numeric_names = {info["name"] for info in numeric_info}

    def current_field(channel: str) -> Optional[str]:
        node = enc.get(channel)
        if isinstance(node, dict):
            return node.get("field")
        return None

    def ensure_numeric(channel: str, exclude: Iterable[str] = ()) -> Optional[str]:
        node = enc.get(channel)
        field = current_field(channel)
        if isinstance(node, dict) and field in numeric_names:
            _apply_axis_format(node, field, df)
            return field

        candidate = _choose_numeric_column(tokens, numeric_info, exclude=exclude)
        if candidate is None:
            return field

        enc[channel] = {
            "field": candidate,
            "type": "quantitative",
        }
        _apply_axis_format(enc[channel], candidate, df)
        return candidate

    def apply_format_if_numeric(channel: str) -> None:
        field = current_field(channel)
        node = enc.get(channel)
        if isinstance(node, dict) and field in numeric_names:
            _apply_axis_format(node, field, df)

    correlation_hint = bool(tokens & {"correlation", "correl", "relationship"})
    scatter_hint = bool(tokens & {"scatter", "scatterplot"})
    vs_hint = bool(tokens & {"vs", "versus"})
    numeric_hint = correlation_hint or scatter_hint or vs_hint

    quantitative_marks = {"point", "line", "area", "trail"}

    if mark_type in quantitative_marks:
        x_field = current_field("x")
        y_field = current_field("y")

        if numeric_hint:
            x_field = ensure_numeric("x")
            y_field = ensure_numeric("y", exclude={x_field} if x_field else set())
        else:
            if x_field is None:
                x_field = ensure_numeric("x")
            else:
                apply_format_if_numeric("x")

            if y_field is None:
                exclude = {x_field} if x_field else set()
                y_field = ensure_numeric("y", exclude=exclude)
            else:
                apply_format_if_numeric("y")

        if numeric_hint:
            x_field = current_field("x")
            y_field = current_field("y")
            if (
                x_field in df.columns
                and y_field in df.columns
                and x_field is not None
                and y_field is not None
            ):
                x_vals = _smart_numeric_series(df[x_field])
                y_vals = _smart_numeric_series(df[y_field])
                sub = pd.concat({x_field: x_vals, y_field: y_vals}, axis=1).dropna()
                if len(sub) >= 2:
                    corr = sub[x_field].corr(sub[y_field], method="pearson")
                    if corr is not None and math.isfinite(corr):
                        title = spec.get("title") or f"{x_field} vs {y_field}"
                        corr_txt = f" (r={corr:.2f})"
                        if corr_txt not in title:
                            spec["title"] = title + corr_txt
    else:
        apply_format_if_numeric("x")
        apply_format_if_numeric("y")

    transforms = spec.get("transform")
    if isinstance(transforms, list):
        cleaned: List[Dict[str, Any]] = []
        for t in transforms:
            if not isinstance(t, dict):
                cleaned.append(t)
                continue
            calc = t.get("calculate")
            if isinstance(calc, str):
                expr = calc.strip().lower()
                if ":" in expr and "datum" not in expr and "parse" in expr:
                    continue
            filt = t.get("filter")
            if isinstance(filt, dict):
                field = filt.get("field")
                if isinstance(field, str):
                    if field not in df.columns:
                        continue
                    if filt.get("operator") == "not null":
                        cleaned.append({"filter": f"datum['{field}'] != null"})
                        continue
            cleaned.append(t)
        if cleaned:
            spec["transform"] = cleaned
        else:
            spec.pop("transform", None)

    return spec
PROFILE_MAX_ROWS = 2000


def _detect_column_role(col_name: str, series: pd.Series, unique_ratio: float) -> str:
    """Infer semantic role of a column based on name and characteristics."""
    name_lower = col_name.lower()

    # Temporal indicators (highest priority)
    if any(word in name_lower for word in ["date", "time", "year", "month", "day", "timestamp"]):
        return "temporal"
    if pd.api.types.is_datetime64_any_dtype(series):
        return "temporal"

    # Geographic indicators
    if any(word in name_lower for word in ["country", "city", "state", "region", "location", "geo", "lat", "lon", "latitude", "longitude"]):
        # But only if not too unique (e.g., coordinates might be high cardinality)
        if unique_ratio < 0.5 or "country" in name_lower or "city" in name_lower or "state" in name_lower:
            return "geographic"

    # Numeric columns
    if pd.api.types.is_numeric_dtype(series):
        # Check for measure keywords first (before ID check)
        if any(word in name_lower for word in ["amount", "price", "value", "revenue", "cost", "sales", "total", "profit", "income", "expense", "valuation", "arr"]):
            return "measure"
        if any(word in name_lower for word in ["count", "quantity", "number", "num", "qty"]):
            return "count"

        # ID/identifier only if very high uniqueness AND name contains "id"
        if "id" in name_lower and unique_ratio > 0.9:
            return "identifier"

        # Otherwise it's a measure (numeric columns are typically measures)
        return "measure"

    # Check if string/object columns are parseable as numeric (e.g., "$1.3B", "1,234.56")
    # This catches financial data stored as formatted strings
    if pd.api.types.is_string_dtype(series) or pd.api.types.is_object_dtype(series):
        # Sample the series for performance
        sample = series.dropna().head(min(100, len(series)))
        if len(sample) > 0:
            parsed = _smart_numeric_series(sample)
            # If >50% of non-null values are parseable as numeric, treat as measure
            parseable_ratio = parsed.notna().sum() / len(sample)
            if parseable_ratio > 0.5:
                # Check for measure keywords
                if any(word in name_lower for word in ["amount", "price", "value", "revenue", "cost", "sales", "total", "profit", "income", "expense", "valuation", "arr"]):
                    return "measure"
                # Even without keywords, if highly parseable, it's likely a measure
                if parseable_ratio > 0.8:
                    return "measure"

    # Categorical indicators (text with low cardinality)
    if unique_ratio < 0.05:
        return "categorical"

    # High uniqueness text columns
    if unique_ratio > 0.9:
        return "identifier"

    return "nominal"


def _suggest_chart_types(numeric_info: List[Dict[str, Any]], categorical_info: List[Dict[str, Any]], has_temporal: bool) -> List[str]:
    """Suggest appropriate chart types based on column characteristics."""
    suggestions = []

    n_numeric = len(numeric_info)
    n_categorical = len(categorical_info)

    # Temporal data → line chart
    if has_temporal:
        suggestions.append("line chart (temporal trends)")

    # 2+ numeric → scatter plot
    if n_numeric >= 2:
        suggestions.append("scatter plot (correlations)")

    # 1 categorical + 1 numeric → bar chart
    if n_categorical >= 1 and n_numeric >= 1:
        suggestions.append("bar chart (category comparisons)")

    # Low cardinality categorical → pie/donut
    low_card_cats = [c for c in categorical_info if c.get("unique", 0) <= 10]
    if low_card_cats:
        suggestions.append("pie/donut chart (part-to-whole)")

    # Many categories → horizontal bar or treemap
    high_card_cats = [c for c in categorical_info if c.get("unique", 0) > 10]
    if high_card_cats:
        suggestions.append("horizontal bar chart (many categories)")

    # Distribution analysis
    if n_numeric >= 1:
        suggestions.append("histogram (distribution)")

    return suggestions[:4]  # Limit to top 4 suggestions


def dataset_profile(
    df: pd.DataFrame,
    *,
    max_cols: int = 30,
    include_quants: bool = True,
    include_viz_hints: bool = True,
) -> Dict[str, Any]:
    """
    Compact, LLM-friendly profile of the current table with visualization hints.
    Keep it small: truncate long strings and cap lists.
    """
    nrows = len(df)
    work_df = df
    if nrows > PROFILE_MAX_ROWS:
        # sample deterministically so repeated calls stay stable and fast
        work_df = df.sample(PROFILE_MAX_ROWS, random_state=0)
    work_rows = len(work_df)
    cols = []
    has_temporal = False
    numeric_cols_info = []
    categorical_cols_info = []

    for i, c in enumerate(df.columns):
        if i >= max_cols:
            break
        s = df[c]
        s_sample = work_df[c] if c in work_df.columns else s
        dtype = str(s.dtype)
        missing = int(s.isna().sum())
        unique = int(s_sample.nunique(dropna=True))
        unique_ratio = unique / nrows if nrows > 0 else 0.0

        info: Dict[str, Any] = {
            "name": c,
            "dtype": dtype,
            "missing_pct": _pct(missing, nrows),
            "unique": unique,
            "unique_ratio": round(unique_ratio, 3),
            "examples": _example_values(s_sample, 3),
        }

        # Detect semantic role
        if include_viz_hints:
            role = _detect_column_role(c, s, unique_ratio)
            info["role"] = role
            if role == "temporal":
                has_temporal = True

        # Check if column is numeric OR parseable as numeric
        s_num = _smart_numeric_series(s_sample)
        parseable_count = s_num.notna().sum()
        is_parseable_numeric = (parseable_count / work_rows) > 0.5 if work_rows > 0 else False

        if pd.api.types.is_numeric_dtype(s) or is_parseable_numeric:
            numeric_cols_info.append({"name": c, "variance": float(s_num.var(skipna=True)) if len(s_num.dropna()) >= 2 else 0.0})
            stats = {
                "min": float(s_num.min()) if work_rows and parseable_count > 0 else None,
                "max": float(s_num.max()) if work_rows and parseable_count > 0 else None,
                "mean": float(s_num.mean()) if work_rows and parseable_count > 0 else None,
                "std": float(s_num.std()) if work_rows and parseable_count > 0 else None,
            }
            if include_quants and parseable_count > 0:
                qs = s_num.quantile([0.25, 0.5, 0.75]).to_dict()
                stats.update(
                    {
                        f"q{int(q * 100)}": float(v)
                        for q, v in qs.items()
                        if not math.isnan(v)
                    }
                )
            info["num_stats"] = stats
            # Add note if it's a string-encoded numeric for LLM context
            if not pd.api.types.is_numeric_dtype(s) and is_parseable_numeric:
                info["note"] = "string-encoded numeric (will be parsed)"
        elif pd.api.types.is_datetime64_any_dtype(s):
            has_temporal = True
            s_dt = pd.to_datetime(s_sample, errors="coerce")
            info["datetime_range"] = {
                "min": s_dt.min().isoformat() if s_dt.notna().any() else None,
                "max": s_dt.max().isoformat() if s_dt.notna().any() else None,
            }
        else:
            categorical_cols_info.append({"name": c, "unique": unique, "ratio": unique_ratio})
            info["top_values"] = _topk_counts(s_sample, 5)

        cols.append(info)

    profile: Dict[str, Any] = {"row_count": nrows, "columns": cols}

    # Add visualization hints
    if include_viz_hints:
        viz_hints = {
            "suggested_chart_types": _suggest_chart_types(numeric_cols_info, categorical_cols_info, has_temporal),
            "summary": {
                "numeric_columns": len(numeric_cols_info),
                "categorical_columns": len(categorical_cols_info),
                "has_temporal_data": has_temporal,
                "total_columns": len(cols),
            }
        }
        profile["visualization_hints"] = viz_hints

    # add 1–3 sample rows (stringified and truncated) for extra context
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
        df[c] = _smart_numeric_series(df[c])


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
                x_vals = _smart_numeric_series(result[x])
                y_vals = _smart_numeric_series(result[y])
                result = result[(x_vals > 0) & (y_vals > 0)]
                result[x] = np.log(_smart_numeric_series(result[x]))
                result[y] = np.log(_smart_numeric_series(result[y]))

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
        if kind == "temporal":
            converted = _maybe_coerce_year_temporal(out[field])
            if converted is not None:
                out[field] = converted
            continue
        if kind == "quantitative":
            out[field] = _smart_numeric_series(out[field])
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
    prompt_lc = prompt.lower()
    prompt_tokens = _tokenize(prompt)
    # pick a table
    if meta_store:
        # keep meta store in sync with live tables to avoid stale entries
        stale = [name for name in meta_store.keys() if name not in tables]
        for name in stale:
            meta_store.pop(name, None)

    table_name, df = pick_table(tables, meta_store)

    numeric_info, categorical_info = _collect_column_stats(df)

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
    auto_ops: List[Dict[str, Any]] = []
    auto_table = False
    if not ops:
        auto_ops, auto_table = _auto_operations(prompt_tokens, categorical_info)
        if auto_ops:
            ops = _validate_ops(auto_ops)

    spec = plan.get("vega_lite")

    tokens = set(prompt_tokens)
    table_tokens = {"table", "tabular", "list"}
    chart_tokens = {
        "chart",
        "graph",
        "plot",
        "pie",
        "bar",
        "line",
        "scatter",
        "map",
    }

    table_requested = bool(tokens & table_tokens)
    chart_requested = bool(tokens & chart_tokens)

    if "not" in tokens and "table" in tokens:
        table_requested = False

    if auto_table and not chart_requested:
        table_requested = True
        if not auto_ops:
            inferred_ops, _ = _auto_operations(prompt_tokens, categorical_info)
            if inferred_ops:
                ops = _validate_ops(inferred_ops)

    if table_requested and not chart_requested:
        if not auto_ops:
            inferred_ops, _ = _auto_operations(prompt_tokens, categorical_info)
            if inferred_ops:
                ops = _validate_ops(inferred_ops)
        spec = None
        vtype = "table"

    # run backend ops to produce the data used by either table or chart
    data = exec_operations(df, ops)

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
    spec = _enforce_prompt_preferences(
        spec,
        prompt_tokens,
        df,
        numeric_info=numeric_info,
        categorical_info=categorical_info,
    )

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

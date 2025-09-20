import re
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple

def normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    # keep original names but add lowercase alias map as needed
    df.columns = [c for c in df.columns]
    return df

def guess_table(tables: Dict[str, pd.DataFrame]) -> Tuple[str, pd.DataFrame]:
    # Use the biggest table by rows as default
    if not tables: 
        raise ValueError("No tables uploaded yet.")
    name = max(tables, key=lambda k: len(tables[k]))
    return name, tables[name]

def vega_pie(values, category="industry"):
    return {
        "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
        "title": f"Breakdown by {category}",
        "data": {"values": values.to_dict(orient="records")},
        "mark": {"type": "arc", "outerRadius": 100},
        "encoding": {
            "theta": {"field": "n", "type": "quantitative"},
            "color": {"field": category, "type": "nominal"}
        },
        "view": {"stroke": None}
    }

def vega_scatter(values, x, y, color=None, title=None, log=True):
    enc = {
        "x": {"field": x, "type": "quantitative"},
        "y": {"field": y, "type": "quantitative"},
        "tooltip": [{"field": x}, {"field": y}]
    }
    if color: enc["color"] = {"field": color, "type": "nominal"}
    spec = {
        "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
        "title": title or f"{y} vs {x}",
        "data": {"values": values.to_dict(orient="records")},
        "mark": {"type": "point", "opacity": 0.7},
        "encoding": enc
    }
    if log:
        spec["encoding"]["x"]["scale"] = {"type": "log"}
        spec["encoding"]["y"]["scale"] = {"type": "log"}
    return spec

def table_top_investors(df: pd.DataFrame, col="investors") -> pd.DataFrame:
    if col not in df.columns:
        # try fuzzy matches
        candidates = [c for c in df.columns if "investor" in c.lower()]
        if candidates: col = candidates[0]
        else: raise ValueError("No investors column found.")
    s = df[col].fillna("").astype(str).str.split(r"[;,]")
    exploded = df.assign(investor=s).explode("investor")
    cleaned = (exploded["investor"].astype(str).str.strip()
               .str.replace(r"\s+", " ", regex=True).str.replace(r"[·•,]", "", regex=True)
               .str.replace(r"\.$", "", regex=True).str.casefold())
    out = (cleaned[cleaned.ne("")].value_counts()
           .reset_index())
    out.columns = ["investor", "count"]
    return out

def corr_arr_valuation(df: pd.DataFrame) -> Tuple[pd.DataFrame, float, float]:
    # find columns approximately
    def find_col(name):
        opts = [c for c in df.columns if name in c.lower()]
        if not opts: return None
        # prioritize exact or simplest
        return sorted(opts, key=len)[0]
    arr_col = find_col("arr") or find_col("annual") or "arr"
    val_col = find_col("valuat") or "valuation"
    if arr_col not in df.columns or val_col not in df.columns:
        raise ValueError("ARR or Valuation column not found.")
    sub = df[[arr_col, val_col]].copy()
    sub = sub.replace([np.inf, -np.inf], np.nan).dropna()
    pearson = float(sub[arr_col].corr(sub[val_col], method="pearson"))
    spearman = float(sub[arr_col].corr(sub[val_col], method="spearman"))
    # include company/industry if present for tooltip
    for extra in ["company", "company_name", "name", "industry"]:
        for c in df.columns:
            if extra in c.lower():
                sub[c] = df[c]
    return sub, pearson, spearman

def route_prompt(prompt: str, tables: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
    name, df = guess_table(tables)

    p = prompt.lower()

    # Easy: pie by industry
    if "pie" in p and ("industry" in p or "breakdown" in p):
        col = next((c for c in df.columns if "industry" in c.lower()), None)
        if not col: raise ValueError("No industry column found.")
        agg = df[col].value_counts(dropna=False).reset_index()
        agg.columns = [col, "n"]
        return {"type": "chart", "title": "Industry Breakdown", "spec": vega_pie(agg.rename(columns={col:"industry"}))}

    # Scatter founded year vs valuation
    if "scatter" in p and ("founded" in p and "valuation" in p):
        fy = next((c for c in df.columns if "founded" in c.lower()), None)
        val = next((c for c in df.columns if "valuat" in c.lower()), None)
        if not (fy and val): raise ValueError("Need founded year and valuation columns.")
        vals = df[[fy, val] + [c for c in df.columns if "industry" in c.lower() or "company" in c.lower()]].dropna()
        return {"type":"chart","title":"Founded Year vs Valuation","spec": vega_scatter(vals.rename(columns={fy:"founded_year", val:"valuation"}), "founded_year","valuation", color=next((c for c in vals.columns if "industry" in c.lower()), None), log=False)}

    # Hard: top investors table
    if "investor" in p and ("most" in p or "frequen" in p or "top" in p):
        tbl = table_top_investors(df)
        return {"type":"table","title":"Top Investors","table":{"columns": list(tbl.columns), "rows": tbl.to_dict(orient="records")}}

    # Extreme: correlation ARR vs Valuation
    if ("arr" in p and "valuation" in p) and ("correlation" in p or "best" in p):
        sub, pear, spear = corr_arr_valuation(df)
        spec = vega_scatter(sub, x=sub.columns[0], y=sub.columns[1], color=next((c for c in sub.columns if "industry" in c.lower()), None), title=f"ARR vs Valuation (r={pear:.2f}, ρ={spear:.2f})", log=True)
        return {"type":"chart","title":"ARR vs Valuation","spec": spec}

    # Fallback: simple guess — list columns
    cols = ", ".join(df.columns)
    raise ValueError(f"Couldn't understand the prompt. Available columns: {cols}")

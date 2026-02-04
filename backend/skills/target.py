"""
Target-aware analysis skills.

Automatic target detection + distribution analysis + feature associations.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from core.models import (
    ColumnRole,
    DataProfile,
    TargetAssociation,
    TargetInsights,
    TargetSpec,
)
from core.utils import smart_numeric_series, tokenize


MAX_ASSOC_ROWS = 5000
MAX_ASSOC_FEATURES = 30


# ---------------------------------------------------------------------------
# Target inference
# ---------------------------------------------------------------------------

_TARGET_KEYWORDS = {
    "target", "label", "class", "y", "outcome", "response", "dependent",
    "price", "sales", "revenue", "churn", "default", "fraud", "score", "rank",
}

_RANK_KEYWORDS = {"rank", "score", "priority"}


def infer_task_and_target(profile: DataProfile) -> TargetSpec:
    """Infer likely target column and task type from profile heuristics."""
    if not profile.columns:
        return TargetSpec(column=None, task_type="unknown", confidence=0.0, reason="empty profile")

    best: Tuple[float, Optional[str], str] = (0.0, None, "unknown")
    reasons: Dict[str, str] = {}

    for col in profile.columns[:MAX_ASSOC_FEATURES]:
        name = col.name
        tokens = set(tokenize(name))
        unique_ratio = (col.cardinality / profile.row_count) if profile.row_count else 0.0

        score = 0.0
        task_type = "unknown"

        if tokens & _TARGET_KEYWORDS:
            score += 0.6
            reasons[name] = "name keyword"

        if tokens & _RANK_KEYWORDS:
            score += 0.2
            task_type = "ranking"

        # Heuristic task type by dtype/cardinality
        is_numeric = "int" in col.dtype or "float" in col.dtype
        if is_numeric:
            if col.cardinality <= 20 or unique_ratio <= 0.05:
                score += 0.15
                task_type = "classification"
            else:
                score += 0.15
                task_type = "regression"
        else:
            if unique_ratio <= 0.1:
                score += 0.2
                task_type = "classification"

        if col.role == ColumnRole.identifier:
            score -= 0.4

        if score > best[0]:
            best = (score, name, task_type)

    confidence = max(0.0, min(1.0, best[0]))
    if confidence < 0.4 or not best[1]:
        return TargetSpec(column=None, task_type="unknown", confidence=confidence, reason="no strong target signal")

    reason = reasons.get(best[1], "heuristic")
    return TargetSpec(column=best[1], task_type=best[2], confidence=confidence, reason=reason)


# ---------------------------------------------------------------------------
# Distribution analysis
# ---------------------------------------------------------------------------

def analyze_target_distribution(df: pd.DataFrame, target: TargetSpec) -> Dict[str, Any]:
    """Analyze target distribution and return warnings + summary."""
    if not target.column or target.column not in df.columns:
        return {"summary": {}, "warnings": ["Target column missing in dataframe."]}

    s = df[target.column]
    warnings: List[str] = []
    summary: Dict[str, Any] = {}

    if target.task_type == "classification":
        vc = s.astype("string").value_counts(dropna=False)
        total = int(vc.sum())
        top = vc.head(10).to_dict()
        summary["class_counts"] = {str(k): int(v) for k, v in top.items()}
        if total > 0:
            max_share = float(vc.iloc[0]) / total
            min_share = float(vc.iloc[-1]) / total
            summary["max_class_share"] = round(max_share, 4)
            summary["min_class_share"] = round(min_share, 4)
            if max_share >= 0.8 or min_share <= 0.05:
                warnings.append("Class imbalance detected; consider stratified split and appropriate metrics.")
    else:
        s_num = smart_numeric_series(s).dropna()
        if s_num.empty:
            return {"summary": {}, "warnings": ["Target has no numeric values."]}

        summary["count"] = int(s_num.shape[0])
        summary["min"] = float(s_num.min())
        summary["max"] = float(s_num.max())
        summary["mean"] = float(s_num.mean())
        summary["std"] = float(s_num.std())
        skew = float(s_num.skew())
        summary["skew"] = round(skew, 4) if math.isfinite(skew) else 0.0
        zero_pct = float((s_num == 0).sum()) / len(s_num)
        summary["zero_pct"] = round(zero_pct, 4)

        if abs(summary["skew"]) >= 1.0:
            warnings.append("Target appears skewed; consider log transform or robust metrics.")
        if zero_pct >= 0.2:
            warnings.append("Target has significant zero inflation.")

    return {"summary": summary, "warnings": warnings}


# ---------------------------------------------------------------------------
# Feature associations
# ---------------------------------------------------------------------------

def _mutual_info_binned(x: pd.Series, y: pd.Series, bins: int = 10) -> float:
    x_bin = pd.qcut(x.rank(method="first"), q=min(bins, x.nunique()), duplicates="drop")
    y_bin = pd.qcut(y.rank(method="first"), q=min(bins, y.nunique()), duplicates="drop")
    ct = pd.crosstab(x_bin, y_bin)
    n = ct.values.sum()
    if n == 0:
        return 0.0
    pxy = ct / n
    px = pxy.sum(axis=1).values.reshape(-1, 1)
    py = pxy.sum(axis=0).values.reshape(1, -1)
    with np.errstate(divide="ignore", invalid="ignore"):
        mi = (pxy * np.log((pxy + 1e-12) / (px * py + 1e-12))).to_numpy()
    return float(np.nansum(mi))


def _cramers_v(table: pd.DataFrame) -> float:
    n = table.values.sum()
    if n == 0:
        return 0.0
    expected = np.outer(table.sum(axis=1), table.sum(axis=0)) / n
    with np.errstate(divide="ignore", invalid="ignore"):
        chi2 = np.nansum(((table - expected) ** 2) / expected)
    r, c = table.shape
    denom = n * max(1, min(r - 1, c - 1))
    return float(math.sqrt(chi2 / denom)) if denom > 0 else 0.0


def _eta_squared(groups: List[pd.Series]) -> float:
    if not groups:
        return 0.0
    means = [g.mean() for g in groups]
    ns = [len(g) for g in groups]
    total_n = sum(ns)
    if total_n == 0:
        return 0.0
    grand_mean = sum(m * n for m, n in zip(means, ns)) / total_n
    ss_between = sum(n * (m - grand_mean) ** 2 for m, n in zip(means, ns))
    ss_total = sum(((g - grand_mean) ** 2).sum() for g in groups)
    return float(ss_between / ss_total) if ss_total > 0 else 0.0


def feature_target_associations(df: pd.DataFrame, profile: DataProfile, target: TargetSpec) -> Tuple[List[TargetAssociation], List[TargetAssociation]]:
    """Compute ranked associations + missingness signals."""
    if not target.column or target.column not in df.columns:
        return [], []

    work = df.sample(min(len(df), MAX_ASSOC_ROWS), random_state=0) if len(df) > MAX_ASSOC_ROWS else df
    target_s = work[target.column]

    assoc: List[TargetAssociation] = []
    miss_assoc: List[TargetAssociation] = []

    # Prepare target series
    if target.task_type == "classification":
        target_c = target_s.astype("string").fillna("NULL")
    else:
        target_c = smart_numeric_series(target_s)

    for col in profile.columns[:MAX_ASSOC_FEATURES]:
        name = col.name
        if name == target.column or name not in work.columns:
            continue
        if col.role == ColumnRole.identifier:
            continue

        s = work[name]
        missing = s.isna().astype(int)

        if target.task_type == "classification":
            # Missingness vs target
            if missing.sum() > 0:
                ct = pd.crosstab(missing, target_c)
                mv = _cramers_v(ct)
                if mv > 0:
                    miss_assoc.append(TargetAssociation(
                        feature=name, metric="missingness_cramers_v", score=round(mv, 4),
                    ))

            if col.role in (ColumnRole.categorical, ColumnRole.nominal, ColumnRole.geographic) or not (
                "int" in col.dtype or "float" in col.dtype
            ):
                ct = pd.crosstab(s.astype("string").fillna("NULL"), target_c)
                v = _cramers_v(ct)
                assoc.append(TargetAssociation(
                    feature=name, metric="cramers_v", score=round(v, 4),
                ))
            else:
                s_num = smart_numeric_series(s)
                groups = []
                for cls, grp in pd.concat([s_num, target_c], axis=1).dropna().groupby(target_c):
                    groups.append(grp.iloc[:, 0])
                eta = _eta_squared(groups)
                assoc.append(TargetAssociation(
                    feature=name, metric="eta_squared", score=round(eta, 4),
                ))
        else:
            s_num = smart_numeric_series(s)
            if s_num.notna().sum() >= 3 and target_c.notna().sum() >= 3:
                aligned = pd.concat([s_num, target_c], axis=1).dropna()
                if aligned.shape[0] >= 3:
                    corr = float(aligned.iloc[:, 0].corr(aligned.iloc[:, 1]))
                    spear = float(aligned.iloc[:, 0].rank().corr(aligned.iloc[:, 1].rank()))
                    mi = _mutual_info_binned(aligned.iloc[:, 0], aligned.iloc[:, 1])
                    assoc.append(TargetAssociation(
                        feature=name,
                        metric="corr_spearman",
                        score=round(abs(spear), 4),
                        direction="positive" if spear >= 0 else "negative",
                        details={"pearson": round(corr, 4), "mi": round(mi, 4)},
                    ))

            # Missingness vs target
            if missing.sum() > 0:
                aligned = pd.concat([missing, target_c], axis=1).dropna()
                if aligned.shape[0] >= 3:
                    mean_miss = aligned[aligned.iloc[:, 0] == 1].iloc[:, 1].mean()
                    mean_full = aligned[aligned.iloc[:, 0] == 0].iloc[:, 1].mean()
                    std = aligned.iloc[:, 1].std() or 1.0
                    effect = (mean_miss - mean_full) / std
                    miss_assoc.append(TargetAssociation(
                        feature=name, metric="missingness_effect", score=round(abs(effect), 4),
                        direction="positive" if effect >= 0 else "negative",
                    ))

    assoc_sorted = sorted(assoc, key=lambda a: a.score, reverse=True)[:10]
    miss_sorted = sorted(miss_assoc, key=lambda a: a.score, reverse=True)[:10]
    return assoc_sorted, miss_sorted


def run_target_analysis(df: pd.DataFrame, profile: DataProfile) -> TargetInsights:
    target = infer_task_and_target(profile)
    if not target.column:
        return TargetInsights(target=target, warnings=["No reliable target detected."])

    dist = analyze_target_distribution(df, target)
    assoc, miss = feature_target_associations(df, profile, target)
    warnings = dist.get("warnings", [])

    return TargetInsights(
        target=target,
        distribution=dist.get("summary", {}),
        associations=assoc,
        missingness=miss,
        warnings=warnings,
    )

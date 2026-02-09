# Auto-EDA Agent Skills Plan

## Goals
- Provide a rich, agent-driven toolset for hypothesis testing, feature engineering, and modeling.
- Keep tools generic and data-driven (no dataset-specific logic).
- Use LLM planner to select tools; tools execute deterministically.

## Tool Categories (Flat Skills)

### Core Modeling
- `linear_regression`
- `logistic_regression`
- `lgbm_regression`
- `lgbm_classification`
- `quantile_regression`

### Time Series
- `seasonality_test`
- `autocorrelation_test`
- `lag_feature_search`
- `rolling_stats`
- `trend_breaks`

### Feature Engineering
- `numeric_transforms`
- `interaction_scan`
- `binning_optimizer`
- `date_part_features`
- `target_encoding`

### Causal-ish / Comparisons
- `group_comparison`
- `matched_comparison`
- `diff_in_diff`
- `uplift_check`

### Diagnostics / Explainability
- `shap_summary`
- `shap_dependence`
- `partial_dependence`
- `leakage_scan`
- `drift_check`
- `missingness_mechanism`

### Agent Planning Skills
- `hypothesis_generator`
- `test_selector`
- `result_validator`

## Implementation Notes
- Planner selects 1-2 tools per query based on intent + profile + context.
- Tools return ViewPlans; builders compute results as tables (or charts when applicable).
- Use LightGBM for non-linear models; use SHAP for explanations.
- Use scikit-learn + statsmodels where useful (matching, tests, quantile regression, ACF/PACF).
- Keep outputs compact and suitable for notebook-style append.

## Dependencies to Add
- `lightgbm`
- `shap`
- `scikit-learn`
- `statsmodels`

## Progress
- Notes created.
- Dependencies: done.
- Implementation: done (tools + builders + planner wiring).

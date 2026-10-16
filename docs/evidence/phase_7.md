# Phase 7 evidence – explanation endpoint

## What was built
`src/fairxai/explain/shap_local.py` (`LocalExplainer`), `/v1/explain` in `serving/app/main.py`. The explainer is built inside `/admin/load`, so a failure to build it fails the load and the slot keeps its previous model. Training now logs `explain_background.csv` (1,000 seeded train rows) for the families that need a background sample; XGBoost needs none.

SHAP values are computed on the encoded features and **summed back onto the 23 raw request features** (the one-hot columns of EDUCATION and MARRIAGE sum to their source column). Summation preserves additivity.

## [VERIFY] shap 0.51.0 with the pinned models
| Family | Explainer | `method` | `output_space` |
| --- | --- | --- | --- |
| XGBoost | `TreeExplainer` (no background) | tree_shap | log_odds |
| RandomForest, DecisionTree | `TreeExplainer(model_output="probability", data=100 bg rows)` | tree_shap | probability |
| LogisticRegression | `LinearExplainer` | linear_shap | log_odds |
| others (NB, KNN, SVM) | `KernelExplainer` on 100 k-means centres, 500 samples, seeded | kernel_shap | probability |

## Live, on the real model (v1, XGBoost) through the gateway (`data/runs/phase7_live.json`)
- `/admin/load` v1 on blue: 200 in 2.1 s (explainer built during load).
- `method=tree_shap`, `output_space=log_odds`.
- **Additivity on 50 eval rows**: base_value + Σ contributions vs logit(probability_default): max abs error **1.25e-6**, mean 2.8e-7 (tolerance 1e-4).
- Example (first eval row, top_k=5): p(default)=0.5459, base 0.0103; PAY_0=-2 → −0.247, BILL_AMT1=0 → +0.188, PAY_AMT3=0 → +0.154, PAY_AMT1=0 → +0.142, PAY_AMT4=0 → +0.136.
- **Latency at 5 requests/s for 60 s (300 requests, `X-Probe`)**: explain p50 28.8 ms, **p95 45.7 ms**, 0 errors (target ≤ 1,000 ms). Predict p95 in the same run: 43.5 ms, 0 errors.
- serving-blue memory with model and explainer: 234 MiB of 1 GiB.

## Tests (`tests/api/test_explain.py`, synthetic fixture models)
Additivity within 1e-4 and a truthful `method`/`output_space` for xgboost, random_forest, decision_tree, logreg and naive_bayes (kernel); all 23 features returned at top_k=23, sorted by |SHAP|; feature values echoed with their types; explain probability equals predict probability; top_k default 10, bounds 1 and 23, rejects 0, 24, -1 and abc with 422; explain is not logged as a prediction; invalid body 422; no key 401; an explainer failure returns 500 with no fallback and no internals leaked; non-XGBoost families refuse to build without a background sample.

Full suite: **236 passed** (`pytest tests`, Postgres tests included).

## Bug found and fixed
Integer feature values were echoed as floats (`PAY_0: -2.0`) because a one-row DataFrame upcasts mixed dtypes. Fixed with `to_dict(orient="records")`, plus a test.

## Exit criteria
- [x] Explain tests pass on the deployed model (v1 via the gateway).
- [x] No stub remains: the 501 path and its test were removed.
- [x] Explain latency recorded.

## Caveat
v1 was trained before `explain_background.csv` was logged. That is harmless for XGBoost, which needs no background, but any non-XGBoost champion must come from a run made after this phase.

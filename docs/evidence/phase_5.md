# Phase 5 evidence – evaluation and fairness gate

## Real-data run
`python -m fairxai.models.train --track`, profile `base`, against MLflow in Docker (Postgres backend, MinIO artifacts at `s3://mlflow/1/e29c476a.../artifacts`). Wall clock 1,416 s.

**Gate decision: `APPROVED`** for XGBoost (the top-ranked tuned model; the selection loop stopped at the first APPROVED model). Registered as `fairxai-credit-default` **version 1**, alias `candidate` set (no `champion` yet; that happens at deployment, Phase 9).

| Gate | Status | Detail |
| --- | --- | --- |
| G1 | PASS | training frame passed suite `batch` |
| G2 | PASS | eval PR-AUC 0.550082 vs floor 0.45 (no champion) |
| G3 | PASS | all six hard checks pass (below) |
| G4 | PASS | no advisory warnings |
| G5 | PASS | temp save/load round-trip OK; registered URI gives identical probabilities on the synthetic smoke set (`g5_smoke=PASS`) |

### Hard fairness checks on the eval partition (4,500 rows, 0 dropped; favorable = predicted not to default)
| Check | Value | Threshold | Status |
| --- | --- | --- | --- |
| eo_sex | 0.027260 | ≤ 0.10 | PASS |
| di_sex | 0.960876 | ≥ 0.80 | PASS |
| eo_age | 0.048388 | ≤ 0.10 | PASS |
| di_age | 0.924184 | ≥ 0.80 | PASS |
| eo_inter (SEX x age_band) | 0.074797 | ≤ 0.15 | PASS |
| di_inter | 0.883824 | ≥ 0.80 | PASS |

Per-group (eval): SEX 1 n=1783 FR 0.7151 TPR_f 0.8092; SEX 2 n=2717 FR 0.7442 TPR_f 0.8364. Age 18-29 n=1465 FR 0.7242; 30-39 n=1684 FR 0.7625; 40+ n=1351 FR 0.7047. Smallest intersectional cell: SEX 1 x 18-29, n=510 (373 with Y=f), all six cells eligible.

### Test partition report (step 6; never read by `decide()`)
3,000 rows at threshold 0.53: PR-AUC 0.5696, F1 0.5631, precision 0.5137, recall 0.6229, accuracy 0.7863. Hard checks on test: eo_sex 0.0417, di_sex 0.9219, eo_age 0.0480, di_age 0.9204, eo_inter 0.0747, di_inter 0.8531 (all would pass).

### Global SHAP (TreeExplainer, 1,000 seeded train rows, log-odds)
Top features by mean |SHAP|: PAY_0 0.406, LIMIT_BAL 0.154, PAY_2 0.144, BILL_AMT1 0.087, PAY_AMT2 0.086, PAY_AMT1 0.086.

### MLflow artifacts on the run
`baselines.csv, config_snapshot.json, fairness_groups.csv, fairness_report.json, gate_report.json, optuna_summary.json, results.json, shap_importance.json, shap_summary.png, test_fairness_report.json`, the logged model, 75 nested trial runs, metrics `fair_*` and `test_*`.

### Version tags and lineage
`gate_decision=APPROVED, decision_threshold=0.53, eval_pr_auc=0.55008, data_version_id=376507f7…, git_commit=86207904…, run_id=e29c476a…, g5_smoke=PASS`. `lineage.resolve(1)` returns every run-level field. **`git_dirty=true`**: the run used uncommitted Phase 5 code. Allowed for a development run; a release deployment requires `git_dirty=false` (D.5), so a clean retrain is needed before release.

## Reproducibility observed on real data
Baseline and tuned CV PR-AUC values are identical across the Phase 3 run and both Phase 5 runs (e.g. xgboost 0.5585, random_forest 0.5537), and the threshold is 0.53 each time.

## Tests
`pytest tests`: **228 passed, 0 skipped** (Postgres tests ran against the live container). New: `tests/unit/fairness/test_fairness.py` (hand-computed EO/DI/DP/ECE, boundaries exactly at 0.10, 0.15 and 0.80, 99-row group, 49 qualified rows, one eligible group, all-same-class group, max FR = 0, NaN probabilities, favorable-label flip, champion-relative warnings, missing group value), `tests/unit/registry/test_promotion.py` (G2 floor and margin boundaries, G2 INDETERMINATE cases, G3 statuses, the full 81-row aggregation truth table, FAIL+INDETERMINATE ⇒ REJECTED, higher PR-AUC but unfair ⇒ REJECTED, G4 never blocks, G1/G5, digest, **Q-08 no hard-coded thresholds**), `tests/unit/models/test_selection.py` (first APPROVED in rank order; none approved ⇒ top-ranked registered with its decision), G5 regression test.

## Incidents during this phase (all fixed, nothing hidden)
1. **G5 bug**: `CategoryRemap` did not set `feature_names_in_`, so the G5 schema check would have failed every real model. Caught while writing the Phase 6 contract test; the in-flight run was stopped before it finished and a regression test was added.
2. **Crash at `end_run`**: MLflow prints the run URL with an emoji; on a Windows cp1252 console this raised `UnicodeEncodeError` after the model had been APPROVED but before registration. That run (`ae5a495f`) was marked `FAILED` with tag `abandoned_reason`; the CLI now reconfigures stdout/stderr to UTF-8. The decision was the same (APPROVED) in both runs.
3. `minio/minio` image no longer available; switched to `cgr.dev/chainguard/minio` (decision log).

## Exit criteria
- [x] Tests pass.
- [x] Fairness report, gate report, SHAP summary and test-set report are in MLflow (MinIO-backed).
- [x] Version tags set; `candidate` alias set only because the decision is APPROVED.
- Risk R-1 did not materialise for this data/model; R-2 (noise on small intersectional cells) remains: the smallest cell has 510 rows.

# Phase 3 evidence – reproducible baseline training

Real run: `python -m fairxai.models.train --profile base` on the UCI Taiwan data (train 18,000 rows, eval 4,500), `PYTHONHASHSEED=0`, 14-core host. Full machine-readable output: [phase_3_results.json](phase_3_results.json) (trial lists omitted).

## Baselines (5-fold CV PR-AUC on train, default class positive)
| Rank | Model | CV PR-AUC |
| --- | --- | --- |
| 1 | random_forest | 0.5353 |
| 2 | xgboost | 0.5117 |
| 3 | svm | 0.5049 |
| 4 | logreg | 0.5013 |
| 5 | knn | 0.4190 |
| 6 | naive_bayes | 0.4061 |
| 7 | decision_tree | 0.2843 |

## Tuned shortlist (Optuna TPE seed 42, 25 trials, 3-fold CV)
| Model | Best CV PR-AUC | Trials | Threshold (OOF F1) | Eval PR-AUC | Eval F1 |
| --- | --- | --- | --- | --- | --- |
| xgboost | 0.5585 | 25 | 0.53 | 0.5501 | 0.5393 |
| random_forest | 0.5537 | 25 | 0.53 | 0.5456 | 0.5421 |
| svm | 0.5148 | 25 | 0.33 | 0.4989 | 0.5205 |

`selected_pending_gate` = xgboost (best CV PR-AUC). **This is not a champion**: the fairness gate (G3) and the PR-AUC floor check run in Phase 5. Eval PR-AUC 0.55 is above the provisional 0.45 floor (N-08), but that is unconfirmed against Stage 2.

## Timing (measured, wall clock)
Total 1130 s (about 19 min): SVM baseline 180 s, SVM tuning 812 s (hit the 900 s per-model bound on trial count/time; all 25 trials completed), random forest tuning 91 s, xgboost tuning 17 s. Peak memory was **not measured**.

## Tests
`pytest tests`: see phase_4.md for the final count. Phase 3 tests: remap equivalence (EDUCATION 5/MARRIAGE 0 == 4/3), unseen category raises, protected-attribute switch, imbalance handling per model, SMOTE refused in production, ranking tie-break, threshold selection, `data_version_id`, and **reproducibility** (two runs on synthetic data under profile `test` agree within 1e-6).

## Not done / caveats
- The spec's "run `FAIRXAI_PROFILE=test make train` twice" on real data was not done; reproducibility is shown only on synthetic data in the unit test (the real run was executed once).
- Peak memory not recorded.
- sklearn 1.9.1 emits a FutureWarning: `SVC(probability=True)` is deprecated (removal in 1.11). It is kept because the spec requires probabilities from SVM; revisit before upgrading sklearn.
- SVM `C`/`gamma` ranges were not given in the spec; chosen as C 0.1-100, gamma 1e-4-1e-1 (log), in `configs/training.yaml`.
- Not a one-hot of unseen EDUCATION values: the encoder raises, by design.

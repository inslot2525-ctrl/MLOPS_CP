# Phase 8 evidence – monitoring and drift detection

## What was built
`src/fairxai/monitoring/{windows,drift,performance,triggers,run}.py`, `src/fairxai/simulator/replay.py` (replay only, no faults), Grafana in compose with provisioned datasources (Prometheus, Postgres) and four dashboards (`infra/grafana/dashboards/{service,model,fairness,lifecycle}.json`), a unique index so re-running monitoring on the same window updates rather than duplicates rows.

## [VERIFY] Evidently 0.7.23 (D7 adapter)
Installs against the pinned stack without changing pandas, numpy or scikit-learn. `Report([DataDriftPreset(num_method="ks", cat_method="chisquare", threshold=0.05)])` returns one `ValueDrift(column=…,method=…)` metric per column whose value is the p-value; `drift.parse_evidently` reads those. **Evidently's `chisquare` is a goodness-of-fit test** of current counts against reference proportions, not a 2-sample contingency test. The scipy contingency path was first written as a contingency test and the engine-agreement test caught the difference (SEX p 0.738 vs 0.794). The fallback now runs the same test, and both engines agree within rel. 1e-3 on every feature. Evidently is the default engine; the scipy path is tested and selectable (`--engine scipy`). No substitution was needed.

## Live run on the real stack (`data/runs/phase8_live.json`)
Model v1 on the blue slot. Stream batches 1–2 replayed through the gateway (4 workers, ≤ 20 requests/s, `request_id = uuid5(seed, batch:row)`); labels of batch n−1 posted during batch n; monitoring run after each batch.

| Window | Predictions | Labels posted | Labelled rows used | data_quality | feature_drift | prediction_drift | performance | fairness | Retrain request |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 (batch 1) | 500 ok / 0 errors, 25.0 s | – | 0 | OK (outlier frac 0.0363) | OK (share 0.00) | OK (rate change 0.021) | **NA** (0 < 300 labelled) | **NA** | no |
| 2 (batch 2) | 500 ok / 0 errors, 25.0 s | batch 1: 500 accepted, 0 dup, 0 unknown | 500 | OK (0.0375) | OK (0.00) | OK (0.003) | OK, PR-AUC **0.5746** (reference 0.5501; ALERT at ≤ 0.5001) | **ALERT** | no |

- Monitoring took 2.7 s and 1.6 s per window. Drift engine: Evidently.
- Re-running monitoring on window 2 left the row count at 10: the upsert is idempotent.
- No retrain request (no drift, performance OK), as expected for unfaulted batches.

### The live fairness ALERT (a real result, recorded, not tuned away)
On window 2's 500 labelled rows, `eo_age` = **0.1555 > 0.10** (FAIL). All other hard checks pass: eo_sex 0.0756, di_sex 0.858, di_age 0.865, eo_inter 0.069, di_inter 0.987. Advisory warnings: dp_sex 0.114, dp_age 0.109, cal_age 0.088. On the 4,500-row eval partition the same model had eo_age 0.048, and 0.048 on test.
This is risk **R-2**: each age band has only about 170 rows in a 500-row window, so the TPR estimates are noisy. Per D.7, a live fairness ALERT sends a notification (logged, `events` row) and **never triggers retraining by itself**; nothing else happened. No threshold was changed. X-14 (bootstrap intervals) is the backlog item that would quantify this.

## Grafana
- Datasource health via API: Prometheus "Successfully queried the Prometheus API"; Postgres "Database Connection OK".
- All four dashboards provisioned (`FairXAI / Service`, `Model`, `Fairness`, `Lifecycle`), no manual clicking.
- Every panel query was executed through `/api/ds/query`: all 19 return HTTP 200. Service, Model and Fairness panels return data. The Lifecycle "Champion version", "Deployments" and "Retraining events" panels return 0 rows **because there are no deployments or retrain events yet** (Phase 9).

## Tests
`tests/unit/monitoring/test_monitoring.py`: streaks with NA in between, drift-only never requests below persistence, 3 drift windows ⇒ request, 2 performance windows ⇒ request, scheduled; no drift on same distribution (both engines); covariate shift detected and engines agree per feature; remap applied before comparison; Evidently stub parsing; threshold fixtures at / below / above for drift share (exactly 0.30 ⇒ ALERT), data quality (299 vs 300 rows, hard failure ⇒ ALERT), prediction-rate change (exactly 0.10 ⇒ WARN), performance (exactly reference − 0.05 ⇒ ALERT, 0.049 ⇒ OK, 299 labelled ⇒ NA, one class ⇒ NA), live fairness min labels.
`tests/unit/monitoring/test_run_db.py` (real Postgres): rows for every signal with NA when unlabelled; same-window rerun idempotent; three shifted windows ⇒ ALERT ×3 and a request only on the third; a new model version resets the streak.
Q-08 test now also scans `monitoring/triggers.py`.
Full suite: **258 passed**.

## Exit criteria
- [x] `monitoring_results` rows for every signal with correct NA handling.
- [x] Grafana dashboards load from provisioning, with data.
- [x] Drift-only fixtures produce no retrain request.

## Caveats
- `make monitor` was not run (no `make` on this host); the job ran through `run_monitoring` (the CLI `python -m fairxai.monitoring.run` uses the same function).
- Retrain *requests* are recorded as notifications/events; acting on them (throttle, export, retrain) is Phase 9.

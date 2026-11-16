# Demo scripts – Person 1 → Person 2 → Person 3

These are word-for-word scripts, using output captured from a full dress rehearsal on the live stack on
2026-11-16. **Say** = spoken lines. **Do** = what to click or type. **Expect** = what the screen shows.
**Deep dive** = extra technical detail for follow-up questions (don't read it out unless asked).
The setup checklist is in [`demo_plan.md`](demo_plan.md) section 0. Run it before Person 1 starts.

Terminal setup (once, Git Bash, repo root):

```bash
source .venv/Scripts/activate; set -a; source .env; set +a
export PYTHONIOENCODING=utf-8 MLFLOW_DISABLE_AGENT_HINT=1
H=(-H "X-API-Key: $SIM_API_KEY" -H "Content-Type: application/json")
A=(-H "X-API-Key: $ADMIN_API_KEY" -H "Content-Type: application/json")
RISKY='{"features": {"LIMIT_BAL": 20000, "SEX": 2, "EDUCATION": 2, "MARRIAGE": 1, "AGE": 24, "PAY_0": 2, "PAY_2": 2, "PAY_3": -1, "PAY_4": -1, "PAY_5": -2, "PAY_6": -2, "BILL_AMT1": 3913, "BILL_AMT2": 3102, "BILL_AMT3": 689, "BILL_AMT4": 0, "BILL_AMT5": 0, "BILL_AMT6": 0, "PAY_AMT1": 0, "PAY_AMT2": 689, "PAY_AMT3": 0, "PAY_AMT4": 0, "PAY_AMT5": 0, "PAY_AMT6": 0}}'
SAFE='{"features": {"LIMIT_BAL": 500000, "SEX": 1, "EDUCATION": 1, "MARRIAGE": 2, "AGE": 35, "PAY_0": -1, "PAY_2": -1, "PAY_3": -1, "PAY_4": -1, "PAY_5": -1, "PAY_6": -1, "BILL_AMT1": 12000, "BILL_AMT2": 9000, "BILL_AMT3": 15000, "BILL_AMT4": 8000, "BILL_AMT5": 7000, "BILL_AMT6": 6000, "PAY_AMT1": 9000, "PAY_AMT2": 15000, "PAY_AMT3": 8000, "PAY_AMT4": 7000, "PAY_AMT5": 6000, "PAY_AMT6": 5000}}'
```

---

# Aryan – BUILD (7 min): problem, architecture, data, training

### 1.1 Opening (1 min) – README on screen

**Say:**
> "Good morning. We built FairXAI: an end-to-end MLOps platform for credit-default prediction. Credit
> scoring is a high-stakes, regulated decision, so the bar is higher than accuracy. The model has to be
> **reproducible**, **fair** across protected groups, and **explainable** for every individual decision,
> and it has to stay that way after deployment. Our core design rule: *no model reaches the registry's
> candidate alias unless it passes a hard fairness gate.* We'll walk the lifecycle in three parts: I'll
> cover Build, [Person 2] covers Trust, and [Person 3] covers Operate."

### 1.2 Architecture (1 min) – README architecture diagram

**Say:**
> "Eight containers in Docker Compose. Data is versioned with DVC, with MinIO as the S3 remote. Training
> logs to MLflow, with Postgres as the backend store and MinIO as the artifact store. Serving is two
> identical FastAPI slots, blue and green, behind our own gateway, which does the cut-over. Every
> prediction is logged to Postgres, labels arrive later through a feedback endpoint, and a monitoring job
> joins them to compute drift, performance and live fairness. Prometheus scrapes the services and Grafana
> shows all of it. Everything is bound to localhost and the API keys are stored only as SHA-256 hashes."

### 1.3 Data versioning and validation (2 min) – terminal

**Do:**
```bash
dvc status
cat dvc.yaml
python scripts/profile_data.py 2>/dev/null | tail -15
```
**Expect:** `Data and pipelines are up to date.` and two stages, `prepare` and `split`. Profile output:
`rows=30000 cols=25 validation_passed=True`, `default_rate=0.2212`, plus the SEX × age-band table.

**Say:**
> "The dataset is UCI's Taiwan credit default data: 30,000 customers, 23 features (credit limit,
> demographics, six months of repayment status, bill amounts and payments), and a 22.1 % default rate.
> `dvc.yaml` defines two stages. Each stage declares its code, config and data as dependencies, so if any
> of them changes, `dvc repro` reruns only what's affected. `dvc status` proves that the data on disk
> matches the hashes committed in `dvc.lock`.
>
> Validation runs before anything else, using Great Expectations: 51 expectations on raw ingestion and
> 50 on the training batch. That covers the schema, value ranges like AGE ≥ 18, allowed category codes, and
> no nulls. Passing the batch suite is gate G1, so a model can't be promoted if its training data
> failed validation.
>
> We look at the subgroups early. This table shows the smallest SEX × age cell is 3,281 rows, which tells
> us intersectional fairness is measurable on this data."

**Deep dive:**
- Splits are seeded and deterministic (seed 42): train 18,000 / eval 4,500 / test 3,000, plus 9 stream
  batches of 500 that simulate production traffic. Same input, same hashes, on any machine.
- `data_version_id` = SHA-256 over the split files. It's tagged on the MLflow run and the model version,
  so you can trace any prediction → model version → run → exact data.
- EDUCATION codes 0/5/6 and MARRIAGE 0 are undocumented in the source data. They're remapped to "other"
  (4 / 3), and a unit test proves the remap is equivalent. An *unseen* category raises an error instead of
  being silently encoded.

### 1.4 Training, tuning and tracking (3 min) – MLflow tab, experiment `fairxai`

**Do:** open the run `e29c476a…` → Metrics, then Parameters, then the nested runs list; show `baselines.csv` in Artifacts.

**Say:**
> "Training is one command, `python -m fairxai.models.train --track`. It runs a three-stage search.
> First, 7 baseline model families with 5-fold stratified CV, scored on **PR-AUC**, not accuracy: with a
> 22 % positive class, a model that never predicts default already gets 78 % accuracy. Random forest led
> the baselines at 0.535 and XGBoost was 0.512.
>
> Second, the top 3 are tuned with Optuna's TPE sampler: 25 trials each, 3-fold CV, a fixed seed and a
> 15-minute budget per model. You can see the 72 nested runs here, each one logged. After tuning, XGBoost
> overtook random forest at 0.5585 CV PR-AUC.
>
> Third, the decision threshold. We don't use 0.5. We sweep 0.05 to 0.95 on **out-of-fold** predictions and
> pick the F1-optimal one, which is 0.53. Choosing it out-of-fold means the eval and test sets never
> influence it.
>
> Results: eval PR-AUC 0.550 and held-out test PR-AUC 0.570. The base rate is 0.22, so that's about 2.5×
> better than random ranking. Test recall is 62 % and precision 51 %.
>
> Reproducibility: we ran the full real training three times and got identical CV scores every time, and
> every run records the git commit, data version and full config snapshot."

**Deep dive:**
- Class imbalance is handled with `class_weight` (`scale_pos_weight` for XGBoost), not SMOTE. SMOTE is
  explicitly refused in the production profile, because synthetic minority rows distort calibration and
  fairness metrics.
- Final XGBoost settings: 398 trees, max_depth 5, learning_rate 0.0104, subsample 0.83, colsample 0.65,
  min_child_weight 2. It's shallow and heavily regularised, a sensible fit for 18k rows.
- Preprocessing lives in the sklearn `Pipeline` that gets logged, so serving runs the same transform code
  as training (no training/serving skew).
- SVM tuning took 812 s of the 1,130 s total; XGBoost tuning took 17 s.

**Handoff:**
> "So we have a reproducible, tracked model with PR-AUC 0.57 on held-out data. But in credit, *accurate*
> isn't enough. [Person 2] will show why this model was allowed into the registry at all."

---

# Rishikesh – TRUST (6 min): fairness gate, registry, serving, SHAP

### 2.1 The promotion gate (2.5 min) – MLflow → run → Artifacts → `gate_report.json`

**Expect** (live artifact):
```json
{"decision": "APPROVED",
 "gates": {"G1": {"status": "PASS", "detail": "suite batch passed"},
           "G2": {"status": "PASS", "detail": "PR-AUC 0.550082 vs floor 0.45"},
           "G3": {"status": "PASS", "detail": {"eo_sex": "PASS", "di_sex": "PASS", "eo_age": "PASS",
                                               "di_age": "PASS", "eo_inter": "PASS", "di_inter": "PASS"}},
           "G4": {"status": "PASS", "detail": []},
           "G5": {"status": "PASS", "detail": "round-trip OK"}},
 "inputs_digest": "d144b9d4…"}
```

**Say:**
> "Every trained model goes through five gates before it can be registered as a candidate. G1 is data
> validation. G2 is performance: PR-AUC must clear a floor of 0.45 and, once a champion exists, beat it
> by at least 0.01. G3 is the fairness gate, and it's **hard**: six checks, all must pass. G4 holds
> advisory checks that warn but never block. G5 is a deployability check: we save the model, reload it
> from the registry URI, and require identical probabilities.
>
> G3 uses two metrics, measured on SEX, age band, and the SEX × age **intersection**. *Equal-opportunity
> difference* is the largest gap in true-positive rate for the favorable outcome (being predicted *not*
> to default) among people who actually didn't default. In other words: are equally creditworthy people
> treated equally? It must be ≤ 0.10, or ≤ 0.15 for the intersection, because those cells are smaller.
> *Disparate impact* is the ratio of approval rates between the worst and best group, and it must be
> ≥ 0.80, the four-fifths rule.
>
> Our values on the 4,500-row eval set: EO 0.027 for sex, 0.048 for age, 0.075 intersectional; DI 0.96,
> 0.92 and 0.88. The held-out test set gives similar numbers. Let me be clear: these are engineering
> tolerances, not proof of legal fairness."

> "One important design point: the gate has **three outcomes**: APPROVED, REJECTED and INDETERMINATE. If a
> group is too small to measure (under 100 rows, or under 50 in the metric's denominator), the
> check is INDETERMINATE, and INDETERMINATE is never promoted. We never treat 'couldn't measure' as
> 'passed'. And a challenger with higher PR-AUC that fails fairness is REJECTED. It's a gate, not a
> weighted trade-off. The aggregation logic is covered by an 81-case truth-table test."

**Deep dive:**
- Selection loop: tuned models are ranked by CV PR-AUC and the **first APPROVED** one is registered. If
  none is approved, the top model is registered with its REJECTED decision, for the audit trail, but it
  never gets an alias.
- `inputs_digest` is a hash of everything the gate saw, so you can verify a decision later.
- Thresholds live in `configs/fairness.yaml` and `configs/gates.yaml`; a test fails if any threshold is
  hard-coded in the source.
- Protected attributes are model inputs (`include_protected: true`). Dropping them doesn't remove proxy
  bias (e.g. LIMIT_BAL), so we gate on outcomes. There's a tested switch to exclude them.

### 2.2 Registry and lineage (0.5 min) – MLflow → Models → `fairxai-credit-default` v1

**Say:**
> "The approved model is registered as version 1 with alias `candidate`. We use MLflow 3 aliases, not the
> deprecated stages. The tags make it auditable on its own: gate decision, threshold 0.53, eval PR-AUC,
> data version hash, git commit and run id."

### 2.3 Live serving (1.5 min) – terminal

**Do:**
```bash
curl -s -i -X POST http://127.0.0.1:8080/v1/predict "${H[@]}" -d "$RISKY" | grep -iE "x-request-id|^\{"
curl -s -X POST http://127.0.0.1:8080/v1/predict "${H[@]}" -d "$SAFE"
```
**Expect:** risky → `"prediction":1,"probability_default":0.8407,"decision_threshold":0.53` with an
`x-request-id` header. Safe → `"prediction":0,"probability_default":0.1419`.

**Say:**
> "Here are two applicants. The first is 24, has a 20k limit, and is two months late on the last two
> bills: 84 % probability of default, above the 0.53 threshold, so predicted to default. The second has a
> 500k limit and always pays duly: 14 %. Every response carries the model version and threshold, so the
> client knows exactly which model decided. The request id is the key that later joins this prediction
> to its true label."

**Do:**
```bash
curl -s -X POST http://127.0.0.1:8080/v1/predict "${H[@]}" -d "${RISKY/\"AGE\": 24/\"AGE\": 7}"
curl -s -X POST http://127.0.0.1:8080/v1/predict -H "X-API-Key: wrong" -H "Content-Type: application/json" -d "$RISKY"
```
**Expect:** `{"error":{"code":"validation_error",…"details":[{"field":"features.AGE","problem":"Input should be greater than or equal to 18"}]}}`
then `{"error":{"code":"unauthorized","message":"missing or invalid API key",…}}`.

**Say:**
> "The input contract is strict. Pydantic validates every field's type and range and rejects unknown
> fields with a 422 that names the exact field. A bad key gets 401, and a client key on an admin route
> gets 403. All errors share one body shape with a request id. Rejected requests are **never** written to
> the prediction log, so monitoring only sees valid traffic. Latency at 5 requests/s through the
> gateway: p95 28 ms for predict, against a 200 ms target."

### 2.4 Explainability (1.5 min) – terminal

**Do:**
```bash
curl -s -X POST "http://127.0.0.1:8080/v1/explain?top_k=5" "${H[@]}" -d "$RISKY"
curl -s -X POST "http://127.0.0.1:8080/v1/explain?top_k=5" "${H[@]}" -d "$SAFE"
```
**Expect:** risky → `PAY_0 +1.119, PAY_2 +0.394, LIMIT_BAL +0.172, PAY_3 −0.131, PAY_AMT1 +0.089`;
safe → `PAY_0 −0.361, LIMIT_BAL −0.244, PAY_AMT2 −0.216, PAY_AMT1 −0.182, BILL_AMT1 −0.140`;
`"method":"tree_shap","output_space":"log_odds","base_value":0.0103`.

**Say:**
> "`/v1/explain` returns per-feature SHAP contributions from TreeSHAP, which computes exact Shapley values
> for tree ensembles in polynomial time, at 43 ms p95. For the risky applicant, being two months late on
> the latest payment, PAY_0, adds +1.12 to the log-odds. That's the single biggest driver. For the safe
> applicant everything pushes down: paid duly, high limit, large repayments.
>
> We report SHAP in **log-odds**, and the response says so explicitly, because that's where the values
> are additive: base value plus contributions equals the model's logit. In probability space they don't
> add up, and that misleads people. This is what a lender needs to give a reason for an adverse decision.
> Globally, PAY_0 is also the top feature (mean |SHAP| 0.41), followed by LIMIT_BAL and PAY_2. That
> matches domain intuition, which is itself a sanity check on the model."

**Deep dive:**
- Explanations come from the model's booster after the pipeline transform, and are mapped back to the
  23 input features (`top_k` 1–23).
- The SHAP summary plot and `shap_importance.json` (1,000 seeded train rows) are logged at training time.

**Handoff:**
> "So the model is fair at release time and every decision is explainable. But data changes after
> deployment. [Person 3] will show how we ship it safely and keep watching it."

---

# Aditya – OPERATE (8 min): blue-green, monitoring, observability, quality

### 3.1 Blue-green deployment (2 min) – terminal

**Do:**
```bash
curl -s http://127.0.0.1:8080/admin/state "${A[@]}"
curl -s -X POST http://127.0.0.1:8080/admin/active "${A[@]}" -d '{"slot":"green"}'
curl -s http://127.0.0.1:8080/v1/model "${H[@]}"
curl -s -X POST http://127.0.0.1:8080/admin/active "${A[@]}" -d '{"slot":"blue"}'
```
**Expect:** state shows blue active and both slots `ready: true, model_version: 1`; the flip returns
`{"active":"green","previous":"blue",…}`; `/v1/model` shows `"slot":"green"`; the flip back returns `"active":"blue"`.

**Say:**
> "Deployment uses blue-green. There are two identical serving slots; a new version is loaded into the
> idle slot through `/admin/load`, which pulls it from the registry and checks it, while the live slot
> keeps serving. Then the gateway switches traffic with one atomic call.
>
> In rehearsal we flipped to green and back **while sending 10 requests a second**: 100 out of 100
> requests succeeded, 0 errors, p95 41 ms. Rollback is the same single call.
>
> The safety rules are what make this production-grade. The gateway refuses to flip to a slot that isn't
> ready and returns 409, so you can't route traffic to an empty slot. If `/admin/load` fails, for example
> on a corrupt artifact, the slot keeps its previous model. And the active slot is persisted in Postgres,
> so a gateway restart doesn't silently flip traffic back."

**Deep dive:**
- Readiness means the model is loaded *and* its version matches what's expected; otherwise `/readyz` returns 503.
- Reloading an already-loaded version takes 0.25 s; a cold load from the registry takes about 3 s.
- Prometheus: `gateway_flips_total` = 4 after rehearsal, and `fairxai_predictions_total` is labelled by slot
  (blue 2,237 / green 62), so you can see exactly where traffic went.

### 3.2 Monitoring – the feedback loop (3.5 min) – terminal, then Grafana Model + Fairness

**Say (before running):**
> "In production, the true labels (did the customer actually default?) arrive late. We simulate that
> with the stream batches. Each batch of 500 goes through the real gateway; the labels for the
> *previous* batch are posted to `/v1/feedback` and joined on request id; then the monitoring job runs
> on the last 500 predictions. Let me run batch 8 live."

**Do** (about 40 s):
```bash
python - <<'EOF'
import os, json, pandas as pd, psycopg
from pathlib import Path
from fairxai.config import load_config
from fairxai.simulator.replay import batch_path, request_ids, send_predictions, send_labels
from fairxai.monitoring.run import model_ref_from_registry, run_monitoring
GW, K, b = "http://127.0.0.1:8080", os.environ["SIM_API_KEY"], 8
df, prev = pd.read_csv(batch_path("data/stream", b)), pd.read_csv(batch_path("data/stream", b - 1))
print(send_predictions(GW, K, df, request_ids(42, b, len(df)), 4, 20))
print(send_labels(GW, K, request_ids(42, b - 1, len(prev)), prev["default"].tolist()))
cfg = load_config(); m = model_ref_from_registry(1, cfg, Path("data/processed/eval.csv"))
with psycopg.connect(os.environ["FAIRXAI_DB_URL"]) as c:
    s = run_monitoring(c, cfg, m, pd.read_csv("data/processed/train.csv"),
                       json.load(open("data/processed/reference_stats.json")))
print(json.dumps(s["signals"], indent=1), s["trigger"])
EOF
```
**Expect:** `{'sent': 500, 'ok': 500, 'non_200': 0}`, `{'accepted': 500, 'duplicates': 0, 'unknown': 0}`,
then five signals and `request: False`. (Batches 1–7 are already in the database; batch 8 is new.
Rehearsals use up batches: if you rehearse again, run batch 9 then, and on the day change `b` to whichever batch is still unused. Do **not** use `make reset`: it deletes all Docker volumes, including the registered model. Re-running an already-used batch is safe: the request ids are deterministic, so the log ignores the repeats, labels show up as `duplicates`, and monitoring recomputes the same window.)

**Say (while it runs):**
> "Five signals per window. **Data quality**: schema and outlier fraction against the training reference.
> **Feature drift**: through Evidently, a Kolmogorov–Smirnov test for numeric features and chi-square for
> categorical, at α = 0.05. The signal is the *share* of drifted features: WARN above 0 and ALERT at 30 %.
> **Prediction drift**: change in the predicted-default rate against the eval distribution.
> **Performance**: PR-AUC on the labelled rows, with ALERT if it falls 0.05 below the reference of 0.55.
> **Live fairness**: the same six G3 checks, on live labelled traffic."

**Show** the rehearsal history (from `monitoring_results`, also on the Grafana Model and Fairness dashboards):

| Window | Feature drift | PR-AUC | Live fairness | Retrain? |
| --- | --- | --- | --- | --- |
| 1 | OK 0.00 | NA (no labels) | NA | no |
| 2 | OK 0.00 | 0.575 | **ALERT** eo_age 0.156 | no |
| 3 | **WARN 0.087** (BILL_AMT1, BILL_AMT2) | 0.515 | WARN (cal_age) | no |
| 4 | OK | 0.508 | WARN (cal_age) | no |
| 5 | OK | 0.581 | WARN (cal_age) | no |
| 6 | OK | 0.611 | WARN (dp_age 0.130); eo_age **0.099** | no |
| 7 | OK | 0.595 | WARN (cal_age 0.052) | no |

**Say:**
> "Three things to note, and they're the most interesting part of the system.
>
> First, window 1 shows **NA**, not OK, for performance and fairness. No labels had arrived yet. Below 300
> labelled rows we refuse to compute the metric, and NA never counts toward a retraining trigger.
>
> Second, window 2 raised a **real fairness ALERT**: the age equal-opportunity gap was 0.156 against a limit
> of 0.10. We didn't tune it away. Each age band has only about 170 rows in a 500-row window, so the TPR
> estimate is noisy. The following windows show 0.04, 0.04, 0.02, and in window 6, 0.099, just under the
> line. That variance is the honest finding. By design, a live fairness ALERT **notifies** a human and
> writes an event, but it never triggers retraining on its own, because retraining on statistical noise
> can make a model less fair. The right fix is bootstrap confidence intervals on the live metrics.
>
> Third, window 3 had 2 of 23 features drifting (both bill amounts) and nothing happened. That's correct.
> Retraining needs **persistence**: 3 consecutive drift ALERT windows or 2 consecutive performance ALERT
> windows, plus a cooldown and a weekly cap. A single noisy window shouldn't spend compute or replace a
> good model."

**Deep dive:**
- Evidently's `chisquare` is a *goodness-of-fit* test against the reference proportions, not a 2-sample
  contingency test. Our scipy fallback first used the contingency test; an engine-agreement test (rel.
  1e-3) caught the difference, and both engines now run the same test.
- Monitoring writes are idempotent: there's a unique index on (kind, model_version, window_end), so
  re-running a window updates the row instead of duplicating it.
- Feedback is idempotent too: a label posted twice counts as a duplicate, and an unknown request id is reported.
- Totals after rehearsal: 3,504 predictions, 3,000 labels, 35 monitoring rows, 0 failed requests.

### 3.3 Observability (1.5 min) – Grafana Service dashboard, then Prometheus /alerts

**Say:**
> "Operational metrics come from Prometheus scraping `/metrics` on the gateway and both slots: request
> counts and latency histograms per route and status, predictions per slot, model-loaded gauges, and log
> failures. The p95 latency computed by Prometheus over the last 5 minutes is 24 ms.
>
> Five alert rules: High5xxRate, HighPredictLatencyP95, SlotDown, ActiveSlotWithoutModel, and
> PredictionLogFailures. That last one matters for ML specifically: if we stop logging predictions,
> monitoring goes blind even though the API looks healthy. All five are inactive right now.
>
> The dashboards are **code**: four JSON dashboards and two data sources (Prometheus and Postgres),
> provisioned when the container starts. Nobody clicks anything to build them, so they're reviewed in git
> like everything else."

### 3.4 Engineering quality and close (1 min)

**Say:**
> "Quality: 258 tests pass against real Postgres and a real local MLflow, with no mocks of the parts that
> matter. They include boundary tests at every fairness threshold, the 81-case gate truth table, the API
> contract, and drift detection. Test fixtures are synthetic, so the real dataset never enters the test
> suite. CI runs lint, format and the tests on every push. Every threshold is in validated YAML, and a
> test fails if one is hard-coded. And every phase has an evidence file of real command output,
> including the incidents we hit and fixed.
>
> Before you ask, the caveats: this model was trained from uncommitted code, so a release build needs a
> clean retrain, and our release policy blocks dirty models; the serving image is 2.4 GB because of the
> scientific stack; and live fairness on 500-row windows is noisy, as you saw.
>
> To sum up: reproducible data and training, a hard fairness gate in front of the registry, explainable
> zero-downtime serving, and monitoring that knows the difference between noise and a real problem.
> Thank you, we'd love your questions."

---

## Live numbers from the dress rehearsal (2026-11-16)

| Item | Value |
| --- | --- |
| `dvc status` | Data and pipelines are up to date |
| Data profile | 30,000 rows, 25 cols, validation passed, default rate 0.2212, AGE 21–79 |
| GE suites | raw_taiwan 51 expectations, batch 50 |
| Gate | APPROVED, G1–G5 PASS, digest d144b9d4… |
| Risky / safe applicant | p = 0.841 → default / p = 0.142 → no default |
| Top SHAP (risky) | PAY_0 +1.119, PAY_2 +0.394, LIMIT_BAL +0.172 |
| Latency @ 5 rps | predict p95 28.3 ms, explain p95 43.2 ms; Prometheus 5 m p95 24 ms |
| Blue-green under 10 rps | 100/100 OK, 0 errors, p95 41 ms |
| Batch 6 monitoring | drift OK, PR-AUC 0.611, fairness WARN (dp_age 0.130), eo_age 0.099 PASS |
| Batch 7 monitoring (tested the Person 3 snippet) | drift OK, PR-AUC 0.595, fairness WARN (cal_age 0.052) |
| Totals | 3,504 predictions, 3,000 labels, 35 monitoring rows |
| Tests | 258 passed |

---

# Tool runbook – how each person runs and explains their tools

Every command assumes the terminal setup at the top of this file (venv active, `.env` loaded, `$H`, `$A`,
`$RISKY` and `$SAFE` defined). All URLs are 127.0.0.1 only. **Live** = safe to run in front of the expert
(seconds). **Don't run live** = slow or changes state; show the output that already exists instead.

## Person 1 – Build tools

### DVC 3.67.1 – data and pipeline versioning
```bash
dvc status                            # Live. "Data and pipelines are up to date." = disk matches dvc.lock
dvc dag                               # Live. ASCII graph: credit_default.csv.dvc -> prepare -> split
cat dvc.yaml                          # Live. Each stage lists its cmd, deps (code + config + data) and outs
cat data/raw/credit_default.csv.dvc   # Live. md5 + size of the raw file; the CSV itself is not in git
dvc repro                             # Don't run live: reruns only stages whose deps changed (none now)
dvc push                              # Don't run live: uploads the cache to the MinIO bucket "dvc"
```
**Explain:** DVC hashes every dependency (data, the code `src/fairxai/data/*.py`, the config
`configs/data.yaml`) and every output; `dvc.lock` commits those hashes to git, so a git commit pins the
exact data. If someone edits `split.py`, `dvc status` shows `split` as changed and `dvc repro` reruns only
that stage. `split` produces train/eval/test, `reference_stats.json` (the baseline for data-quality
monitoring) and `data/stream/` (9 × 500 replay batches), seeded with 42 from `configs/base.yaml`.

### MinIO – S3 object storage
- Console: http://127.0.0.1:9001 (`MINIO_ROOT_USER` / `MINIO_ROOT_PASSWORD` from `.env`).
- Show bucket **`mlflow`** → `1/<run_id>/artifacts/` (gate_report.json, model files) and bucket **`dvc`**.
- **Explain:** MLflow and DVC both talk to it through the S3 API (endpoint URL + `AWS_ACCESS_KEY_ID` /
  `AWS_SECRET_ACCESS_KEY`), so moving to AWS S3 is a change of endpoint and credentials, not of code. The
  image is `cgr.dev/chainguard/minio` because `minio/minio` is no longer pullable (decision log).

### Great Expectations 1.24.0 – data contract
```bash
python scripts/profile_data.py 2>/dev/null | head -4   # Live: rows=30000 cols=25 validation_passed=True, errors: []
python -c "import json;[print(f,len(json.load(open('pipelines/expectations/'+f))['expectations'])) for f in ['raw_taiwan.json','batch.json']]"
pytest tests/data -q                                    # Live, seconds
```
- Suites: `pipelines/expectations/raw_taiwan.json` (51 expectations, raw ingestion) and `batch.json` (50, the
  training frame). Code: `src/fairxai/data/validate.py`, `src/fairxai/data/ingest.py`.
- **Explain:** the expectations cover column set and types, ranges (AGE 18–100, LIMIT_BAL 1–2,000,000, PAY_* −2…9, SEX 1–2), not-null on every column, and other
  allowed codes. The suites are JSON in git, so a contract change is
  code-reviewed. Passing `batch` is **gate G1**, so a model trained on invalid data can't be registered.
  The same contract is enforced online by Pydantic (Person 2): bad data is blocked in training *and* at
  serving time.

### pandas 3.0.6 + scikit-learn 1.9.1 – features and baselines
```bash
sed -n 1,12p configs/training.yaml   # Live: 7 models, cv_folds 5, tuning_cv_folds 3, shortlist 3, optuna_trials 25, class_weight
```
- Code: `src/fairxai/features/pipeline.py` (`build_pipeline(model_name, estimator, include_protected)`) and
  `src/fairxai/models/`. MLflow artifact `baselines.csv` holds the 7 CV PR-AUC scores.
- **Explain:** category remap (EDUCATION 0/5/6 → 4, MARRIAGE 0 → 3) → encoding/scaling → estimator, all in
  one sklearn `Pipeline` that is logged as the model, so serving runs the identical transform (no skew).
  Stratified K-fold keeps the 22 % default rate in every fold. Imbalance is handled with
  `class_weight`/`scale_pos_weight`; SMOTE is refused in production because synthetic rows distort
  calibration and fairness metrics.

### XGBoost 3.2.0 – the selected model
- MLflow run → Parameters: `n_estimators 398, max_depth 5, learning_rate 0.0104, subsample 0.826,
  colsample_bytree 0.654, min_child_weight 2` (histogram tree method).
- **Explain:** gradient boosting fits each new tree to the gradient of the log-loss of the current ensemble.
  Shallow trees, a low learning rate × many trees, and row/column subsampling are the regularisers. The
  model is tree-based, so Person 2's TreeSHAP explanations are exact.

### Optuna 5.0.0 – hyperparameter tuning
- Config: `configs/training.yaml` → `search_spaces` (int/float, log-scale where relevant), `optuna_trials: 25`,
  `optuna_timeout_s: 900`. MLflow: run → 72 nested child runs (sort by metric); artifact `optuna_summary.json`.
- **Explain:** TPE (Tree-structured Parzen Estimator) splits past trials into "good" and "bad", fits a
  density to each, and proposes the next point where good/bad is highest. That's Bayesian optimisation, far
  more sample-efficient than a grid for the same budget. It's seeded (42), runs 3-fold CV inside every
  trial, and has a 900 s cap per model (SVM used 812 s, XGBoost 17 s).

### MLflow Tracking 3.17.1
- UI: http://127.0.0.1:5000 → experiment `fairxai` → run `e29c476a…`: Metrics (`cv_pr_auc`, `eval_*`,
  `test_*`, `fair_*`), Parameters, Tags (`git_commit`, `git_dirty`, `data_version_id`, `dvc_train_md5`), Artifacts.
```bash
python -c "from mlflow import MlflowClient as C; r=C().get_run('e29c476ae99543288e53238281a38196'); print({k:round(v,4) for k,v in r.data.metrics.items() if k.startswith('test_')})"   # Live
```
- Training (**don't run live**, 20–25 min): `python -m fairxai.models.train --track`. Flags: `--profile`,
  `--data-dir`, `--out-dir`; `--track` logs to MLflow and registers the model.
- **Explain:** the tracking server runs in Docker. Its backend store is Postgres (runs, params, metrics) and
  its artifact store is MinIO (`s3://mlflow`). It's started with `--allowed-hosts`, because MLflow 3 rejects
  unknown Host headers (an anti DNS-rebinding check we hit and fixed).

## Person 2 – Trust tools

### MLflow Model Registry
```bash
curl -s "http://127.0.0.1:5000/api/2.0/mlflow/registered-models/search" | python -m json.tool | head -40   # Live
```
- UI: http://127.0.0.1:5000 → Models → `fairxai-credit-default` → Version 1 (alias **`candidate`**, tags).
  Code: `src/fairxai/registry/`.
- **Explain:** the registry decouples *which model* from code. A slot loads by version:
  `/admin/load {"model_version": 1}` → `models:/fairxai-credit-default/1`. Aliases are movable pointers:
  `candidate` is set only when APPROVED, and `champion` marks the deployed model. Version tags carry the
  lineage: `run_id`, `git_commit`, `data_version_id`, `decision_threshold`, `eval_pr_auc`,
  `gate_decision`, `g5_smoke`.

### Fairness gate (our code) – G1–G5
```bash
cat configs/fairness.yaml configs/gates.yaml                                             # Live
pytest tests/unit/fairness/test_fairness.py tests/unit/registry/test_promotion.py -q     # Live, seconds
```
- Code: `src/fairxai/fairness/` (metrics) and `src/fairxai/registry/` (aggregation, promotion). MLflow
  artifacts: `gate_report.json`, `fairness_report.json`, `fairness_groups.csv` (per-group n, favorable
  rate, TPR), `test_fairness_report.json`.
- **Explain the math:**
  - Favorable = predicted *not* to default (ŷ = 0). FR_g = P(ŷ = 0 | g).
  - **Disparate impact** = min_g FR_g / max_g FR_g, must be ≥ 0.80.
  - **Equal-opportunity difference** = max_g TPR_g − min_g TPR_g, where TPR_g = P(ŷ = 0 | y = 0, g); must be
    ≤ 0.10 (sex, age band) and ≤ 0.15 (sex × age band).
  - Eligibility: a group needs n ≥ 100 and ≥ 50 rows in the metric's denominator; otherwise the check is
    INDETERMINATE.
  - Aggregation: any FAIL ⇒ REJECTED; else any INDETERMINATE ⇒ INDETERMINATE; all PASS ⇒ APPROVED.
    That's 3⁴ = 81 status combinations, all in the truth-table test.
  - G4 advisory: demographic-parity difference and calibration gap > 0.05 warn but never block.
  - Worked example (eval, sex): FR 0.7151 vs 0.7442 ⇒ DI = 0.961; TPR 0.8092 vs 0.8364 ⇒ EO = 0.027.

### SHAP 0.51.0 – explanations
```bash
curl -s -X POST "http://127.0.0.1:8080/v1/explain?top_k=5" "${H[@]}" -d "$RISKY"    # Live
curl -s -X POST "http://127.0.0.1:8080/v1/explain?top_k=23" "${H[@]}" -d "$SAFE"    # Live: all 23 features
```
- Code: `src/fairxai/explain/shap_local.py` (per request) and `shap_global.py` (at training time → MLflow
  `shap_summary.png`, `shap_importance.json`).
- **Explain:** TreeExplainer computes exact Shapley values by walking the trees (polynomial time, no
  sampling), so explain p95 is about 43 ms. Additivity: `base_value + Σ shap_value = logit(p)`. Check on the
  risky applicant: logit(0.841) = ln(0.841 / 0.159) = 1.664, and 0.0103 + all 23 contributions = 1.664 (verified live). The
  output is log-odds because that's the only space where the values add up. Global importance = mean |SHAP|
  over 1,000 seeded training rows.

### FastAPI 0.143.2 + Pydantic 2.14.1 + Uvicorn 0.54.0 – serving slots
```bash
curl -s http://127.0.0.1:8001/readyz; echo; curl -s http://127.0.0.1:8002/readyz       # Live: each slot directly
curl -s http://127.0.0.1:8080/v1/model "${H[@]}"                                         # Live
curl -s -X POST http://127.0.0.1:8080/v1/predict "${H[@]}" -d "${RISKY/\"AGE\": 24/\"AGE\": 7}"   # Live: 422
docker exec fairxai-postgres-1 psql -U fairxai -d fairxai -c "select request_id, model_version, slot, probability_default from predictions order by ts desc limit 3;"   # Live
pytest tests/api -q                                                                       # Live, about 1 min
```
- Code: `serving/app/` (`main.py`, `schemas.py`, `model_store.py`) + `serving/Dockerfile`; shared HTTP helpers
  in `src/fairxai/httpapi.py`.
- **Explain:** each slot is a FastAPI app under Uvicorn (ASGI) holding one model in memory (about 207 MiB).
  `schemas.py` builds the request model from the 23 features with types and bounds and
  `ConfigDict(extra="forbid")`, so unknown fields are a 422. Auth compares SHA-256(key) with the stored hashes,
  with separate client and admin keyrings (a client key on an admin route = 403). `X-Probe: true`
  requests are answered but not logged (latency and deploy checks). If the log write fails, the client
  still gets the prediction, flagged `X-Logged: false`, and `fairxai_log_failures_total` increments.
  If the registry can't be reached during `/admin/load`, the result is a 500 and the old model stays loaded.

## Person 3 – Operate tools

### Docker + Compose v2
```bash
C="docker compose --env-file .env -f infra/compose/docker-compose.yml --profile core"
$C ps --format 'table {{.Service}}\t{{.Status}}'   # Live: 8 services
docker stats --no-stream                            # Live: memory per container
$C logs --tail=20 gateway                           # Live
```
- Files: `infra/compose/docker-compose.yml`, `serving/Dockerfile`, `gateway/Dockerfile`, `infra/mlflow/Dockerfile`.
- **Explain:** healthchecks plus `up --wait` give startup ordering (postgres/minio → `init` → mlflow → slots →
  gateway). The one-shot `init` container creates the databases, schema and buckets, and is safe to rerun.
  Ports bind to 127.0.0.1. Memory limits: each slot `mem_limit: 1g`, gateway `256m`. Both slots run the same
  image (`fairxai-serving:${IMAGE_TAG}`), so blue and green differ only by the loaded model.

### Gateway (our FastAPI app) – blue-green
```bash
curl -s http://127.0.0.1:8080/admin/state "${A[@]}"                                    # Live
curl -s -X POST http://127.0.0.1:8080/admin/active "${A[@]}" -d '{"slot":"green"}'     # Live: flip
curl -s -X POST http://127.0.0.1:8080/admin/active "${A[@]}" -d '{"slot":"blue"}'      # Live: flip back
curl -s -X POST http://127.0.0.1:8002/admin/load "${A[@]}" -d '{"model_version": 1}'   # Live: reload a slot
docker exec fairxai-postgres-1 psql -U fairxai -d fairxai -c "select * from gateway_state;"   # Live
```
- Code: `gateway/`; tests `tests/api/test_gateway.py`.
- **Explain:** the gateway proxies `/v1/*` to the active slot with httpx and returns 502 if that slot is
  unreachable. Before a flip it calls the target's `/readyz` and refuses with 409 `slot_not_ready` if it isn't
  ready. The active slot is stored in the `gateway_state` table, so it survives restarts. Metrics:
  `gateway_active_slot`, `gateway_flips_total`, `gateway_upstream_errors_total`,
  `gateway_request_latency_seconds`.

### PostgreSQL 16
```bash
docker exec fairxai-postgres-1 psql -U fairxai -d fairxai -c "\dt"     # Live: 8 tables
docker exec fairxai-postgres-1 psql -U fairxai -d fairxai -c "select count(*) preds, (select count(*) from labels) labels from predictions;"
docker exec fairxai-postgres-1 psql -U fairxai -d fairxai -c "select window_end::time, kind, status, round(value::numeric,3) from monitoring_results order by window_end desc, kind limit 10;"
```
- Schema: `infra/postgres/schema.sql`. Tables: `predictions`, `labels`, `monitoring_results`, `events`,
  `gateway_state`, `deployments`, `retrain_events`, `data_batches`. Databases: `fairxai`, `mlflow`, `airflow`.
- **Explain:** `request_id` is the join key between a prediction and its later label. Replay ids are
  `uuid5(seed, "batch:row")`, so re-sending a batch is idempotent. A unique index on (kind, model_version,
  window_end) turns monitoring re-runs into updates, not duplicates.

### Evidently 0.7.23 – drift and monitoring job
```bash
python -m fairxai.monitoring.run                    # Live (~5 s): last 500 predictions of the newest model version
python -m fairxai.monitoring.run --engine scipy     # Live: same tests, scipy fallback
pytest tests/unit/monitoring -q                     # Live
```
- Code: `src/fairxai/monitoring/{windows,drift,performance,triggers,run}.py`. Config: `configs/monitoring.yaml`,
  `configs/retraining.yaml`.
- **Explain:** `Report([DataDriftPreset(num_method="ks", cat_method="chisquare", threshold=0.05)])`. KS
  statistic = max |F_ref(x) − F_cur(x)| between the empirical CDFs. Chi-square = goodness of fit of the
  current category counts against the reference proportions. Signal = share of drifted features
  (WARN > 0, ALERT ≥ 0.30). Performance = PR-AUC on the labelled rows (NA below 300; ALERT ≤ reference −
  0.05). Triggers: 3 consecutive drift ALERTs or 2 performance ALERTs ⇒ a retrain request, then a 60 min
  cooldown and at most 3 per week. A fairness ALERT only notifies.

### Prometheus v3.5.0 (+ prometheus-client 0.26.0)
- UI: http://127.0.0.1:9090 → Status → Targets (3 up) and Alerts (5 rules). Paste these into the Query tab (Live):
```promql
histogram_quantile(0.95, sum by (le) (rate(gateway_request_latency_seconds_bucket[5m])))
sum by (slot) (fairxai_predictions_total)
gateway_flips_total
fairxai_model_loaded
```
```bash
curl -s http://127.0.0.1:8080/metrics | grep -E "^gateway_(flips|active)"   # Live: raw exposition format
```
- Config: `infra/prometheus/prometheus.yml` (scrape config) and `alerts.yml` (5 rules, `promtool check rules` = SUCCESS).
- **Explain:** Prometheus pulls `/metrics` from each target. Latency is recorded as a histogram, so p95 is
  computed server-side with `histogram_quantile` over a rate window. Each alert rule has a `for:` duration
  (1–5 min), so a single spike doesn't fire it.

### Grafana 12.1.1
```bash
curl -s -u admin:$GRAFANA_ADMIN_PASSWORD "http://127.0.0.1:3000/api/search?type=dash-db" | python -m json.tool | grep title   # Live
```
- UI: http://127.0.0.1:3000 (admin / `GRAFANA_ADMIN_PASSWORD`) → Dashboards → folder **FairXAI**:
  **Service** (request rate, p95, 5xx, active slot), **Model** (prediction rate, drift share, PR-AUC per
  window), **Fairness** (each check per window against its threshold). Skip **Lifecycle**: it has no data
  in this setup. Set the time range to "Today".
- Files: `infra/grafana/provisioning/` (datasources and dashboard provider) and
  `infra/grafana/dashboards/{service,model,fairness,lifecycle}.json`.
- **Explain:** everything is provisioned at container start from files in git. The Postgres datasource runs
  SQL on `monitoring_results` for the ML panels; the Prometheus datasource runs PromQL for the service panels.

### CI and tests – GitHub Actions, ruff 0.17.1, pytest 9.1.1
```bash
ruff check . && ruff format --check . && python scripts/check_tree.py   # Live, seconds
export FAIRXAI_TEST_DB_URL="postgresql://fairxai:${POSTGRES_PASSWORD}@127.0.0.1:5432/fairxai"
pytest tests/unit/monitoring -q       # Live alternative, seconds
pytest tests -q                       # Don't run live (~4 min); quote "258 passed" from the rehearsal
```
- Workflow: `.github/workflows/ci.yml` (`lint-test` job on every push); show the green check on GitHub.
- **Explain:** `check_tree.py` enforces the repo layout, and a test fails on any hard-coded threshold. DB
  tests run against real Postgres when `FAIRXAI_TEST_DB_URL` is set; otherwise they're reported as
  *skipped*, never as passed. Registry tests use a real local MLflow (sqlite). Fixtures are synthetic
  (`tests/fixtures/make_synthetic.py`).

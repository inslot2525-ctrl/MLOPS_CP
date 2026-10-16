# FairXAI MLOps – Audited Implementation Specification

Oct 15, 2026 · @hello

## Summary

The source spec (FairXAI MLOps Technical Specification v1.0 draft) is a sound skeleton but is **not implementation-ready**: it ends after Section 18, references a missing Open Parameters Register (Section 22) and Section 23, leaves 28 parameters undefined, and describes about 16 containers, two data-versioning systems and shadow plus blue-green deployment, which is too much for a course project. This document replaces it as the single source of truth for coding agents.

**What is verified.** Only the two files supplied were inspected: the master prompt and the v1.0 spec. The FairXAI brief, the MLOps brief, the Stage 1 and Stage 2 code, and the datasets themselves were **not available**. Dataset facts below come from the public UCI documentation of the Taiwan credit-default data and are marked *unverified against your copy*. No test result, benchmark or hardware figure in this document is measured.

**What changed (all ratifiable, each with a fallback in Section C).**

- DVC only for data versioning; LakeFS removed. Batch isolation is done with a quarantine directory and a hashed data-version manifest.
- FastAPI replaces BentoML. A small gateway service provides the blue-green flip and replaces the reverse proxy plus Docker-socket control.
- Blue-green only; the shadow stage becomes an extension. One Compose project instead of three environments.
- Alertmanager, counterfactual and individual fairness, audit logging and the separate protected-attribute table move to the extension backlog.
- Hard fairness gates (Equal Opportunity, Disparate Impact, Intersectional) are fully defined; the held-out data is split four ways so that tuning, gating and reporting never share rows.

**Decisions only you can make or confirm** (the project proceeds on the stated defaults until you do):

1. Provide the Stage 1 and Stage 2 code so the preprocessing contract can be diffed against it (N-02, blocking for the claim "reproduces Stage 1" only).
2. Confirm the favorable outcome is *not predicted to default* (N-03) and the fairness tolerances and age bands (N-04, N-05).
3. Confirm the host machine's CPU and RAM (O-28) and the deadline and team capacity; neither was supplied, so the schedule is relative.
4. Ratify or reject the seven scope changes listed as CR-1 to CR-7 in Section C.

**Known risk that no document can remove:** it is not established that any model on this data passes the hard fairness gates. If none does, the system correctly refuses to promote and the owner must decide what to do (Section A, finding A-07).

## A. Audit report

**Scope of the audit.** Inspected: the master prompt and the v1.0 spec (all 18 sections present in the file). Not inspected because not supplied: both briefs, Stage 1 and 2 code, datasets, any diagrams or appendices. The audit is therefore complete for the supplied files only.

| ID | Sev. | Finding | Practical impact | Resolution in this document |
| --- | --- | --- | --- | --- |
| A-01 | Critical | The spec ends at Section 18. Section 22 (Open Parameters Register) and Section 23 (Integrated evaluation) are cited but absent. All 28 open items O-01 to O-28 appear only as inline mentions with no definition. | Open items cannot be traced to their original wording; an agent would have to guess. | Every O-item was reconstructed from context and given a status in Section C. Reconstructions are labelled; none is presented as original text. |
| A-02 | Critical | The FairXAI brief, MLOps brief, Stage 1 and 2 code and datasets are cited as authority ("brief Section 7", "deliverable 13") but unavailable. | Cannot confirm no brief requirement was dropped; column contract unverified. | Brief-derived rules kept as written. Dataset facts taken from public UCI documentation and flagged. Phase 2 profiles the real file and diffs it against Stage 1 when supplied. |
| A-03 | High | D22 stores protected attributes in a separate access-controlled table, but SEX and AGE are almost certainly model inputs, so they travel in every request and in the logged features. | The separation gives no protection and adds a table, a join and an audit log. | Single `predictions` table; separate table, access control and audit log moved to extensions (CR-6). |
| A-04 | High | The evaluation partition is used both to choose the champion (D21) and to gate it (G2, G3), and the test set is never named. | Selection leakage: reported fairness and performance are optimistic. | Four-way split: train, eval (selection and gates), test (reporting only), stream (simulation). N-06. |
| A-05 | High | G6 (shadow), G8 (approval) and CD depend on O-01, O-02 and O-11, which are unresolved and partly circular (approval needs an environment; environments need a CD target; CD cannot reach local Compose). | Sections 10, 12 and 16 cannot be implemented. | One Compose project plus an ephemeral CI stack; shadow removed; approval defined as a CLI step (default off) plus a GitHub Environment for the release job. N-14, N-17, N-18. |
| A-06 | High | DVC and LakeFS both retained (D3) on shared MinIO, plus Postgres databases for each tool. | Two extra services, hooks and failure modes for a capability (isolated validate-then-merge) that a manifest provides. | LakeFS removed; quarantine directory plus data-version manifest (CR-1). |
| A-07 | High | No evidence that any trained model passes the three hard fairness gates, and no rule for "no model passes". | The initial champion could be impossible to produce; the pipeline would stall silently. | Gate returns `REJECTED` or `INDETERMINATE`; the pipeline stops, notifies and never bypasses. Threshold-adjustment mitigation listed as extension X-09. Documented as risk R-1. |
| A-08 | High | The favorable outcome (O-06) is undefined, yet it changes every ratio and difference. | Equal Opportunity and Disparate Impact could be computed on the wrong class. | N-03 fixes it: favorable = predicted *not* to default. Configurable. |
| A-09 | Medium | Retraining compares candidate and champion on a fixed evaluation partition, which cannot reflect a concept shift. | A genuinely better candidate would not look better, or a worse one would not look worse. | Evaluation set for retraining = fixed eval partition plus a 30 percent holdout of newly labelled rows. N-10. |
| A-10 | Medium | The Taiwan data has no timestamp; drift windows cannot be time-based. | Drift results depend on batch ordering. | Windows are defined by prediction-log order (last W rows). Limitation recorded in the model card. |
| A-11 | Medium | Tool versions unpinned (O-26). Great Expectations, Evidently, Airflow and MLflow changed APIs substantially across recent releases. | Code written against remembered APIs may not run. | Lockfile strategy N-01; every version-sensitive call is isolated in one adapter module with a \[VERIFY\] test. Evidently has a scipy fallback. |
| A-12 | Medium | About 16 containers (postgres, minio, lakefs, mlflow, 2 airflow, 3 serving, proxy, prometheus, alertmanager, grafana, simulator, plus an unchosen exporter). Host specs unknown (O-28). | Likely to exceed a student laptop; slow startup. | 10 persistent services in the MVP (Section B). Estimates only; Phase 0 measures. |
| A-13 | Medium | Three of seven fairness dimensions (calibration, counterfactual, individual) are custom code with no validated reference; the flip test is meaningless if protected attributes are not model inputs. | Unvalidated code could produce plausible but wrong warnings. | Calibration kept (small, unit-tested); counterfactual and individual moved to extensions X-05, X-06. |
| A-14 | Medium | Validation requires outlier detection and quality-over-time checks without numeric limits. | Untestable acceptance criteria. | Limits defined in D.2 (range rules from the public contract, IQR-based outlier warning, per-batch quality rows). |
| A-15 | Medium | Container resource monitoring needs an exporter (cAdvisor or similar) not chosen. | Extra service for an optional signal. | Dropped (X-10). Monitoring results reach Grafana via Postgres, which also resolves O-15. |
| A-16 | Medium | Airflow would need Docker control to deploy and flip (O-02), which implies mounting the Docker socket. | Privilege escalation risk and a fragile coupling. | Gateway admin API and an idle-slot `/admin/load` endpoint; no Docker socket. N-18. |
| A-17 | Medium | Stage 3 (class weight vs SMOTE) and Stage 6 (cross-dataset) are research tasks, but the production pipeline depends on their results (`imbalance_strategy`, O-22). | Production work blocked on unfinished research. | Default `class_weight`; SMOTE and cross-dataset run as separate experiments (N-21, N-22). |
| A-18 | Low | Terminology drift: "lifecycle state" vs MLflow stages (deprecated in favour of aliases), "Bento tag" vs model version, "evaluation partition" vs "held-out stream". | Ambiguity for agents. | One glossary in D.1; aliases `candidate`, `champion`, `previous_champion`. |
| A-19 | Low | Section 18 says pass criteria are "written alongside the test". | Circular; not checkable in advance. | Section H lists measurable criteria per requirement. |

**Architectural risks that remain after simplification.**

- R-1: the fairness gate may reject every candidate (A-07).
- R-2: point-estimate fairness gates on about 4,500 evaluation rows are noisy for small intersectional cells; a model near a threshold can flip between runs. The minimum-size rule and reported intervals mitigate but do not remove this.
- R-3: SVM with probability outputs is slow on 18,000 rows; if shortlisted, tuning is bounded by a per-model time limit.
- R-4: the scripted concept-shift scenario cannot guarantee that retraining beats the champion. The demo therefore asserts decision logic with forced fixtures and asserts only pipeline completion for the organic scenario (Section D.9).

**Compatibility concerns to verify at build time (none verified here).** Airflow import paths (TaskFlow decorators moved between major versions); MLflow alias API; Great Expectations 1.x checkpoint API; Evidently report result dictionary; Trivy action inputs; GitHub Environment required reviewers on private repositories.

## B. Final architecture

The MVP runs **10 persistent containers** plus one on-demand simulator job. RAM figures are rough estimates from typical idle footprints and are **not measured**; Phase 0 must measure them.

### B.1 Component inventory

| Component | Responsibility (lifecycle stage) | Inputs → outputs | Depends on | Class | Est. RAM | If removed |
| --- | --- | --- | --- | --- | --- | --- |
| postgres | One server, three databases: `airflow`, `mlflow`, `fairxai` (prediction log, monitoring, events) | SQL | none | Essential | 0.2 GB | Everything stateful stops |
| minio | S3 store: bucket `mlflow` (artifacts), bucket `dvc` (data remote) | S3 | none | Essential (cheap; keeps the cloud path) | 0.2 GB | Replace with local volumes; loses S3 parity |
| mlflow | Tracking and model registry (train, register, promote) | HTTP → runs, versions, aliases | postgres, minio | Essential | 0.4 GB | No tracking or registry |
| airflow-webserver, airflow-scheduler | Orchestration of 7 DAGs (all stages); LocalExecutor | DAG triggers → task runs | postgres | Essential (D2) | 1.5–2 GB | No automation; pipelines become scripts |
| gateway | Blue-green router: forwards `/v1/*` to the active slot, flips on admin request, exposes `/metrics` | HTTP → HTTP | serving slots | Essential (replaces proxy plus Docker control) | 0.1 GB | No flip, no rollback |
| serving-blue, serving-green | Identical FastAPI containers; each loads one model version on request | `/v1/predict`, `/v1/explain`, `/v1/feedback`, `/admin/load` | mlflow, minio, postgres | Essential | 0.4 GB each | No serving |
| prometheus | Scrapes slots and gateway; evaluates alert rules (service health) | `/metrics` → time series | slots, gateway | Essential | 0.2 GB | No service-health signal |
| grafana | Dashboards over Prometheus and Postgres | queries → panels | prometheus, postgres | Essential | 0.2 GB | No dashboards (a demo requirement) |
| simulator (profile `sim`) | Replays stream batches, injects faults, sends labels | CSV → HTTP | gateway | Essential for demo only | job | No reproducible demo |
| DVC (CLI in Airflow and dev images) | Versions raw data and splits; remote on MinIO | files → pointers, hashes | minio, git | Essential (D3) | none | No data versioning |
| Great Expectations (library) | Validates raw data and every ingested batch | DataFrame → pass or fail plus report | none | Essential (D6) | none | No quality gate |
| Evidently (library) | Drift computation behind one adapter | two DataFrames → drift dataclass | none | Essential (D7) with scipy fallback | none | Hand-written drift only |
| Fairlearn, SHAP, Optuna (libraries) | Group metrics; explanations; tuning | arrays → metrics, values, best params | none | Essential (D12, D18, D21) | none | See D |
| GitHub Actions, gitleaks, pip-audit, Trivy | CI, scanning, ephemeral integration stack, release job | git push → pass or fail | repo | Essential (D9) | n/a | No CI/CD |
| *Removed or optional:* LakeFS, shadow slot, Alertmanager, BentoML, cAdvisor, NannyML, Vault, Docker secrets | See Section C (CR-1 to CR-7) and the extension backlog |  |  | Optional or redundant |  |  |

**Compose profiles.** `core` = the 10 services above (default). `sim` = simulator. `alerting` = Alertmanager (extension X-03). Minimum configuration for development without Airflow: run `core` minus the two Airflow services and call the `src` CLIs directly (`make train`, `make deploy`); every DAG task is a thin call to such a CLI.

A one-shot `init` container (not counted among the 10) creates the MinIO buckets and the three Postgres databases, applies `infra/postgres/schema.sql`, and exits.

### B.2 System architecture

```text
                       +----------------- GitHub Actions (CI, scans, ephemeral stack, release) -----------------+
                                                      | builds images, runs tests
  data/raw (DVC) ---> [airflow-scheduler/webserver] --+--> src/ CLIs (train, gate, deploy, monitor, retrain)
                          |        |        |
               MLflow API |        | SQL    | admin API
                          v        v        v
   +--------+      +---------+  +----------+   +-----------------------------+
   | minio  |<---->| mlflow  |  | postgres |   | gateway  (active = blue)    |
   +--------+      +---------+  +----------+   +-------+---------------+-----+
        ^  artifacts     ^ load model             | /v1/*            | /v1/*(idle: no traffic)
        |                |  (admin/load)     +----v-----+       +-----v----+
        +----------------+------------------>| serving- |       | serving- |
                                              | blue     |       | green    |
   simulator --HTTP--> gateway                +----+-----+       +-----+----+
                                                   | /metrics          | /metrics
                          prometheus <-------------+-------------------+----- gateway /metrics
                              |
                          grafana <---- postgres (monitoring_results, deployments, retrain_events)
```

### B.3 Data flow

```text
raw CSV --dvc--> data/raw --prepare--> data/processed --split--> train | eval | test | stream batches (data/stream/batch_NNNN.csv)
                                                          |
train+eval --> training DAG --> MLflow (runs, model version, reports)
stream batch --> simulator --> gateway --> slot --> predictions table (features, prediction, probability, model_version)
simulator (delayed) --> /v1/feedback --> labels table
monitoring DAG: reads predictions (+labels) --> monitoring_results --> Grafana, retrain decision
retraining DAG: labelled rows --> data/ingest/incoming --GE--> accepted | rejected --> new data_version_id --> training pipeline
```

### B.4 Training and promotion flow

```text
validate data (G1) -> 5-fold CV of 7 baselines on train -> rank by PR-AUC -> shortlist top 3 -> Optuna (TPE, seeded)
 -> pick best by CV PR-AUC -> fit on full train -> choose decision threshold from out-of-fold predictions (N-07)
 -> score eval set -> fairness report -> decide() = APPROVED | REJECTED | INDETERMINATE   (G1-G5)
 -> always register version with tags; alias `candidate` only if APPROVED
 -> REJECTED / INDETERMINATE: champion unchanged, event + notification
```

### B.5 Monitoring and retraining flow

```text
monitoring DAG (triggered per window)
  data-quality (GE on window) -> drift (Evidently adapter) -> prediction drift -> performance (labelled rows only) -> fairness (labelled rows)
  -> write monitoring_results
  -> evaluate triggers:  drift only            => WARN event, no retrain
                         drift persists N=3    => retrain request
                         perf degraded x2      => retrain request
                         scheduled + enough data => retrain request
  -> retraining DAG (max_active_runs=1, unique RUNNING row):
       throttle check -> export labelled rows -> GE -> train -> gates -> deploy DAG
```

### B.6 Deployment and recovery flow

```text
read alias `candidate` -> idle slot = other than gateway.active -> POST idle/admin/load {version}
   load fails -> idle slot keeps previous model; deployment FAILED; champion untouched
 -> G6 live-slot verification (200 smoke rows through idle slot directly)
   fails -> FAILED; champion untouched
 -> [G8 approval if enabled]
 -> gateway POST /admin/active {slot: idle}   (flip)
 -> G7 observation (>= 100 requests and >= 120 s, max 600 s): 5xx <= 1 %, p95 <= 500 ms
   pass -> aliases: previous_champion <- champion, champion <- candidate; deployment DEPLOYED
   fail -> gateway flips back; aliases unchanged; deployment ROLLED_BACK; alert
Manual rollback: load `previous_champion` into the idle slot, then flip (same checks).
```

Every diagram above matches the contracts in Section D.

## C. Authoritative decision register

**Status vocabulary.** `RESOLVED` = supported by the source spec or verified evidence. `DEFAULTED` = a concrete value chosen by documented engineering judgment (it may supersede part of an original decision; those cases are listed as change requests CR-1 to CR-7 and need owner ratification, with the original behaviour as fallback). `BLOCKING` = needs owner input. `OPTIONAL` = deferrable. Original identifiers D1 to D23 and O-01 to O-28 are preserved; new decisions are N-01 to N-28; extensions are X-01 to X-15.

### C.1 Original decisions D1–D23

| ID | Topic | Final value | Status | Rationale; rejected alternatives | Impact and dependencies | Verified by |
| --- | --- | --- | --- | --- | --- | --- |
| D1 | Runtime | Docker Compose, profiles `core`, `sim`, `alerting`. Kubernetes only as a documented path (`docs/cloud-path.md`, one page). | RESOLVED | Local-first per brief. Kubernetes manifests rejected as scope. | `infra/compose/` | `make up` brings `core` healthy |
| D2 | Orchestration | Airflow, LocalExecutor, Postgres metadata DB, TaskFlow API, 7 DAGs (D.8). Version chosen in Phase 0 from Airflow's official constraints file for Python 3.11. | RESOLVED | Required by D2. Celery/Kubernetes executors rejected (more services). | Import path for `@dag`/`@task` differs by major version \[VERIFY\] | DAG import test in CI |
| D3 | Data versioning | **DVC only.** Remote = MinIO bucket `dvc`. Batch isolation = quarantine directories plus a hashed manifest (D.2). | DEFAULTED (CR-1) | LakeFS adds 1 service, 1 database, hooks; its merge semantics are reproduced by `data_batches` status plus manifest. Rejected: both tools (A-06). **Fallback if owner rejects:** add X-02 after Phase 4. | `data_version_id` replaces the LakeFS commit ID everywhere | Lineage test (H) |
| D4 | Object storage | MinIO, buckets `mlflow` and `dvc` | RESOLVED | S3 parity for the cloud path at \~0.2 GB. Local volumes rejected only for parity. | Credentials via `.env` | `mc ls` in smoke test |
| D5 | Tracking and registry | MLflow, Postgres backend, MinIO artifacts; registry **aliases** `candidate`, `champion`, `previous_champion` (stages are deprecated) | RESOLVED | Aliases are the supported mechanism in current MLflow \[VERIFY on pinned version\]. | `registry/` module is the only caller of alias APIs | Alias unit test, lineage test |
| D6 | Validation | Great Expectations, wrapped by `data/validate.py`; suites in `pipelines/expectations/` | RESOLVED | Required. Pandera/custom rejected (decided). | GE 1.x API \[VERIFY\]; only `validate.py` imports GE | Valid and invalid fixture tests |
| D7 | Drift | Evidently behind `monitoring/drift.py::compute_drift()` returning our dataclass. **Contingency:** if the pinned Evidently cannot return per-feature test results within a 1-day timebox, implement KS (numeric) and chi-square (categorical) with scipy, and record the substitution. | RESOLVED | Isolates API churn. | Thresholds in `configs/monitoring.yaml` | Drift fixture tests |
| D8 | Serving | **FastAPI** (uvicorn). Same five properties as the spec lists for BentoML: hashed-key auth, typed validation, liveness and readiness, `/metrics`, structured logs. | DEFAULTED (CR-2) | All five are native or one-liners in FastAPI; BentoML needs custom middleware for most and its API changed across 1.x releases. **Fallback:** X-08 swaps the framework without changing any contract in D.6. | "Bento tag" becomes image tag and model version | API contract tests |
| D9 | CI/CD | GitHub Actions; CI on every push and PR; `release` job in GitHub Environment `production` with required reviewer \[VERIFY availability for the repo's visibility and plan; fallback `workflow_dispatch` with a typed confirmation\] | RESOLVED | Required. Other CI rejected. | No remote deploy target (N-18) | Workflow run evidence |
| D10 | Promotion gate | Deterministic `decide()` over G1–G8 (D.4) | RESOLVED |  | `registry/promotion.py` | Gate pass/fail tests |
| D11 | Fairness gating | Hard: Equal Opportunity, Disparate Impact, Intersectional on SEX and age bands. Tracked and warned in MVP: Demographic Parity, Calibration. Counterfactual and Individual Fairness **moved to X-05, X-06**. MARRIAGE and EDUCATION reported, not gating. | DEFAULTED for the move (CR-5); RESOLVED for the gate set | Custom code without a validated reference (A-13). | `fairness/` | Metric unit tests against hand-computed fixtures |
| D12 | Fairness library | Fairlearn `MetricFrame` for group metrics | RESOLVED |  | Version pinned | Metric tests |
| D13 | Retraining | Tiered triggers, persistence windows, cooldown, weekly cap; values in N-10 | RESOLVED (values DEFAULTED) |  | `configs/retraining.yaml` | Throttle and idempotency tests |
| D14 | Deployment | **Blue-green with automatic rollback.** Shadow stage removed (X-01). | DEFAULTED (CR-3) | Shadow and blue-green together need mirrored traffic, a comparison job and a third slot with no required behaviour that blue-green plus live-slot verification (G6) does not demonstrate. Rejected: keeping both. **Fallback:** X-01. | G6 redefined (N-13) | Deployment scenario tests |
| D15 | Demo traffic | Simulator replays stream batches; faults: covariate shift, concept shift, corrupted batch; plus forced bad candidates (D.9) | RESOLVED |  | `simulator/` | `make demo` |
| D16 | Monitoring | Prometheus and Grafana in `core`; **Alertmanager moved to profile `alerting`** (X-03). Prometheus alert rules still evaluate and are visible in its UI. | DEFAULTED (CR-4) | Alertmanager's only job is webhook routing, which `notify()` already does for pipeline events. | `infra/prometheus/alerts.yml` | Rule unit test with `promtool` |
| D17 | Notifications | `notify()` writes the `events` table and logs; optional webhook via `NOTIFY_WEBHOOK_URL` and `NOTIFY_WEBHOOK_STYLE=slack\|discord` (blank = off) | DEFAULTED | Resolves O-19 without needing a webhook account. | `src/fairxai/notify.py` | Unit test with a local HTTP stub |
| D18 | Explanations | `/v1/explain` on the loaded model; global SHAP summary logged to MLflow per version | RESOLVED |  | D.6, N-23 | Explain tests |
| D19 | Secrets | `.env` (git-ignored) plus committed `.env.example`; gitleaks in CI and pre-commit; API keys stored as SHA-256 hashes. **Docker secrets and Vault not implemented**; Vault is one paragraph in `docs/cloud-path.md`. | DEFAULTED (CR-7) | Proportionate to a local course project. Never hardcode secrets. |  | gitleaks CI job; test that no key is returned or logged |
| D20 | Repository | Monorepo, top-level folders per the spec (Section E) | RESOLVED |  |  | Tree check script |
| D21 | Model selection | 7 baselines ranked by 5-fold CV PR-AUC; shortlist **3**; Optuna; champion = best CV PR-AUC among tuned models that pass the gate, evaluated on the eval partition | RESOLVED (shortlist size DEFAULTED to 3) |  | N-11 | Selection test |
| D22 | Prediction logging | One `predictions` table (features as JSON, which include SEX and AGE when they are model inputs), `labels` table, 90-day purge job. **Separate protected table, access control and audit log moved to X-07.** | DEFAULTED (CR-6) | A separate table is meaningless when protected attributes are model inputs (A-03). | `fairxai` database | Purge test |
| D23 | Rollback | (1) Gateway flip back within the observation window. (2) Reload `previous_champion` into the idle slot, then flip. Image redeploys are *code deployment* done by CD, not rollback. | DEFAULTED (CR-7) | Keeps both mechanisms; removes Docker control. | D.8 | Rollback scenario tests |

**Change requests for owner ratification.** CR-1 (D3 LakeFS), CR-2 (D8 FastAPI), CR-3 (D14 shadow), CR-4 (D16 Alertmanager), CR-5 (D11 two metrics), CR-6 (D22 table and audit), CR-7 (D19 secrets and D23 wording). Until ratified, the specification in Section D is authoritative; if any CR is rejected, the stated fallback is added as a separate phase after the MVP exit, never mixed into it.

### C.2 New decisions N-01 to N-28

All values live in `configs/`; the key is given where it matters. "Alt." lists the rejected alternative.

| ID | Topic | Final value | Status | Resolves | Rationale; alt. | Verified by |
| --- | --- | --- | --- | --- | --- | --- |
| N-01 | Python and locking | Python 3.11. `uv pip compile` produces `requirements/*.lock` (app, airflow, serving, dev). Airflow installed with its official constraints file. No dependency upgrade without a decision. | DEFAULTED | O-26 | Stable across the stack. Alt.: Poetry (extra tool). | CI installs from lock files only |
| N-02 | Taiwan data contract | See D.2: 30,000 rows; `ID`, 23 features, target `default`. Public-documentation facts, **unverified against the team's file**. Stage 1 equivalence check requires the Stage 1 code. | BLOCKING for the claim "reproduces Stage 1"; MVP proceeds on the public contract | O-03 | Cannot be resolved without the file and code. | `scripts/profile_data.py` output reviewed; diff vs Stage 1 when supplied |
| N-03 | Favorable outcome | Favorable = predicted **not** to default (Ŷ = 0). Config `fairness.favorable_label: 0`. | DEFAULTED, owner confirmation requested | O-06 | In credit scoring the harm is wrongful denial. Alt.: default = positive for fairness (reverses Disparate Impact meaning). | Unit test flips the key and checks results change as hand-computed |
| N-04 | Age bands | `18-29`, `30-39`, `40+` (label `age_band`). Config `fairness.age_bands`. | DEFAULTED, owner confirmation requested | O-07 | Three bands keep SEX x age cells large enough. Alt.: four bands (small 50+ cells). Band sizes measured in Phase 2. | Band edge tests at 29/30 and 39/40 |
| N-05 | Fairness tolerances | Equal Opportunity difference ≤ 0.10; Disparate Impact ratio ≥ 0.80 (min group rate / max group rate, for any number of groups); Intersectional (SEX x age\_band): EO difference ≤ 0.15 and DI ≥ 0.80. Group eligibility: n ≥ 100 rows **and** ≥ 50 rows in the metric's denominator. Warn margins: DP diff > 0.10 or calibration gap > 0.05, and worse than champion by > 0.02. | DEFAULTED, owner confirmation requested | O-08 | 0.80 is the conventional four-fifths rule; 0.10/0.15 are conventional engineering tolerances, **not legal standards**; passing does not show the model is fair. | Boundary tests at exactly the threshold |
| N-06 | Splits and seeds | Stratified on target x SEX, seed 42: train 60 % (18,000), eval 15 % (4,500), test 10 % (3,000), stream 15 % (4,500 = 9 batches of 500, kept in file order). Test set used only for the final report. | DEFAULTED | O-04 | Fixes A-04. Alt.: 3-way split (leakage). | Split disjointness and stratification tests |
| N-07 | Decision threshold | One global threshold per model, chosen on out-of-fold training predictions to maximise F1 of the default class; stored as `decision_threshold`. No group-specific thresholds. | DEFAULTED | O-05 | Deterministic, no use of eval or test. Alt.: fixed 0.5 (poor under 22 % prevalence). | Test that eval rows are never used |
| N-08 | Performance gate | Primary metric PR-AUC of the default class on the evaluation set. Challenger must exceed champion by **≥ 0.01** (retraining). Initial champion: PR-AUC ≥ **0.45** (floor; unverified against Stage 2, must be revisited when Stage 2 results are supplied). Accuracy alone never decides. | DEFAULTED | O-09 | Sampling noise of PR-AUC on 4,500 rows is about 0.01–0.02, so a smaller margin would promote noise. | Gate tests |
| N-09 | Monitoring parameters | Reference = train partition. Window = last 500 logged predictions (`window_rows`), requires ≥ 300. Per-feature drift: KS (numeric) or chi-square (categorical), p < 0.05. Dataset drift when ≥ 30 % of features drift. Prediction drift on probability (KS) and positive rate. Performance window requires ≥ 300 labelled rows; degradation = PR-AUC ≥ 0.05 below the champion's eval PR-AUC. Monitoring DAG is triggered per window (schedule `None`; optional `@hourly`). | DEFAULTED | O-14 | Conventional tests; thresholds tunable. | Fixture tests for each rule |
| N-10 | Retraining policy | Request when: dataset drift in **3** consecutive windows; **or** performance degradation in **2** consecutive labelled windows; **or** scheduled run with ≥ 1,000 new labelled rows. Cooldown 60 min, max 3 runs per rolling 7 days (demo profile: cooldown 5 min, cap 3). Exported labelled rows are consumed once (marked in the export transaction, released only if the event ends FAILED), so after a REJECTED or INDETERMINATE candidate a new retrain needs at least 500 new unused labelled rows. Any trigger additionally needs at least 500 unused labelled rows (scheduled: 1,000). Retraining reuses the champion's tuned hyperparameters (no re-tuning). Evaluation set = fixed eval partition + 30 % holdout of new labelled rows (chosen by hash of `request_id`); both models scored on the same set. New rows weighted `new_data_weight: 3`. | DEFAULTED | O-16 | Fixes A-09 and loops. | Throttle, idempotency, concurrency tests |
| N-11 | Tuning | 5-fold stratified CV for ranking; Optuna TPE seed 42, 25 trials, 3-fold CV, 900 s limit per model; Optuna storage in memory, trials logged to MLflow. | DEFAULTED | O-25 | Bounded runtime. | Run-time check in Phase 3 |
| N-12 | Serving targets | **Targets, not guarantees:** predict p95 ≤ 200 ms and explain p95 ≤ 1,000 ms at 5 requests/s on the dev machine; container memory limit 1 GB. Measured in Phase 6, recorded, never invented. | DEFAULTED | O-13 |  | Load smoke test records actuals |
| N-13 | G6 and G7 | G6 live-slot verification: 200 eval rows through the idle slot, 0 errors, p95 ≤ 500 ms. G7 post-flip: ≥ 100 requests and ≥ 120 s (max 600 s), 5xx rate ≤ 1 %, p95 ≤ 500 ms. | DEFAULTED | O-12 |  | Deployment scenarios |
| N-14 | Shadow | Not in MVP (X-01). G6 as above. | DEFAULTED | O-11 | CR-3 |  |
| N-15 | API keys | Header `X-API-Key`. `API_KEYS=client:sha256hex,...`; `ADMIN_API_KEY_SHA256`. Constant-time compare. Keys never logged or echoed. | DEFAULTED | D8, D19 |  | Auth tests |
| N-16 | Retention | 90 days for `predictions` and `labels`; daily purge DAG. Audit log not implemented (X-07). | DEFAULTED | O-23 |  | Purge test |
| N-17 | Environments | One Compose project `fairxai` for local use. CI runs the same files plus `docker-compose.ci.yml` as an ephemeral stack. No dev/staging/production stacks. | DEFAULTED | O-01 |  | CI integration job |
| N-18 | Deployment control and CD | Airflow talks to the gateway and slots over HTTP only (no Docker socket). CD = ephemeral integration stack on the GitHub-hosted runner, then `release` job (environment `production`) that builds and tags images. No push by default; GHCR push optional (X-11). | DEFAULTED | O-02, O-20 | A hosted runner cannot reach a local stack. | Workflow review |
| N-19 | Notifications | See D17. | DEFAULTED | O-19 |  |  |
| N-20 | CI tooling | Ruff (lint and format), pytest, pip-audit, gitleaks, Trivy image scan (fail on fixable HIGH or CRITICAL; exceptions only in a reviewed `.trivyignore`). | DEFAULTED | O-27 |  | CI log |
| N-21 | Imbalance strategy | `imbalance_strategy: class_weight` until the Stage 3 result is recorded. KNN and Naive Bayes have no class weights and run unweighted; the threshold step (N-07) compensates. SMOTE runs only inside CV training folds in the Stage 3 experiment script. | DEFAULTED | A-17 |  | Leakage test for SMOTE experiment |
| N-22 | Cross-dataset | German Credit and any real-world dataset are run by `experiments/cross_dataset/` outside production DAGs. Real-world dataset not selected: **OPTIONAL**. | OPTIONAL | O-22 |  |  |
| N-23 | Explanation method | TreeExplainer for tree models, LinearExplainer for logistic regression, KernelExplainer (100 k-means background rows) for others. Response states `method` and `output_space` truthfully. | DEFAULTED | D18 |  | Additivity test: contributions + base = model output within 1e-4 |
| N-24 | Missing data | The API rejects any request with a missing or null feature (422); it never imputes. Training data with missing values fails validation (G1). Fairness groups with missing group or label rows exclude those rows and report the count. | DEFAULTED | O-03 |  | Tests |
| N-25 | Ingest and labels | New data enters by triggering the `ingest_batch` DAG (CLI, REST or simulator). Labels return through `POST /v1/feedback`. | DEFAULTED | O-17, O-18 |  | Scenario tests |
| N-26 | Human approval | `approval.model_promotion: false` by default; when true, a deployment stops at `AWAITING_APPROVAL` until `make approve VERSION=n`. CD approval is the GitHub Environment (D9). | DEFAULTED | G8, O-01 |  | Both modes tested |
| N-27 | Reproducibility | Same code, config, data and machine give identical selected hyperparameters and metrics within 1e-6; XGBoost `n_jobs` fixed in config; cross-machine equality is not claimed. | DEFAULTED |  | Unavoidable nondeterminism: thread scheduling, library versions. | Reproducibility test |
| N-28 | Hardware | Unknown. Provisional budget: 4 CPU cores, 16 GB RAM, 20 GB disk; minimum attempt 8 GB with Airflow only. **Measured values must be recorded in `docs/resource-report.md` in Phase 0 and Phase 11.** | BLOCKING for any compatibility claim | O-28 |  | Report exists with real numbers |

### C.3 Open-item mapping, blockers and extension backlog

**Every O-item and where it is resolved.** O-01 → N-17, N-26. O-02 → N-18. O-03 → N-02, N-24 (BLOCKING for the Stage 1 claim only). O-04 → N-06. O-05 → N-07. O-06 → N-03. O-07 → N-04. O-08 → N-05. O-09 → N-08. O-10 → D5. O-11 → N-14. O-12, O-13 → N-13, N-12. O-14 → N-09. O-15 → B.1 (Postgres to Grafana; no exporter). O-16 → N-10. O-17, O-18 → N-25. O-19 → D17. O-20 → N-18. O-21 → B.1 (one Postgres, three databases). O-22 → N-22. O-23 → N-16. O-24 → D.9. O-25 → N-11. O-26 → N-01. O-27 → N-20. O-28 → N-28 (BLOCKING for compatibility claims).

**Blocking items (owner input needed).**

| ID | Missing information | Consequence while missing |
| --- | --- | --- |
| N-02 | Stage 1 preprocessing code, Stage 2 results, the actual dataset file | MVP uses the public contract. The claim "reproduces Stage 1" and the N-08 floor cannot be confirmed. |
| N-28 | Host CPU, RAM, disk | No claim that the stack fits. Phase 0 measures. |
| Planning | Deadline and team capacity | Schedule is relative (Section F). A feature freeze is defined as a phase boundary, not a date. |

**Extension backlog** (the MVP is fully functional without all of it). Relative complexity: S small, M medium, L large.

| ID | Extension | Prerequisites | Size | Attempt only when |
| --- | --- | --- | --- | --- |
| X-01 | Shadow traffic stage | Phase 11 done; gateway mirroring | M | MVP demo recorded and owner wants it |
| X-02 | LakeFS data lake (reverses CR-1) | Phase 4 | L | Owner rejects CR-1 |
| X-03 | Alertmanager and webhook routing | Prometheus rules | S | Webhook account exists |
| X-04 | Real-world dataset and German Credit in the lifecycle | Dataset selected | M | After Stage 6 experiments |
| X-05 | Counterfactual flip test | Protected attributes are model inputs | S | Fairness gate stable |
| X-06 | Individual fairness (nearest-neighbour consistency) | Similarity definition chosen | M | After X-05 |
| X-07 | Separate protected-attribute table, access control, audit log | Roles in Postgres | M | Owner rejects CR-6 |
| X-08 | BentoML serving (reverses CR-2) | Phase 6 contracts frozen | M | Owner rejects CR-2 |
| X-09 | Fairlearn ThresholdOptimizer mitigation | Risk R-1 materialises | M | No model passes the gate |
| X-10 | cAdvisor container metrics | Prometheus | S | Hardware report needs it |
| X-11 | Self-hosted runner deploy or GHCR push | A reachable host | M | Remote deployment required |
| X-12 | NannyML performance estimation without labels | Monitoring DAG | M | Labels unavailable |
| X-13 | Docker secrets or Vault | Cloud move | M | Leaving local use |
| X-14 | Bootstrap confidence intervals on fairness metrics | Gate stable | S | R-2 matters in the report |
| X-15 | Additional serving replicas | Gateway load balancing | M | Load requirement appears |

**Feature freeze.** After Phase 10 exit, no extension work may begin until Phase 12's demonstration has been recorded.

## D. Implementation specification

This section is the engineering contract. Where it differs from the source spec, Section C says why. Component names use the Python package `fairxai` under `src/fairxai/`.

### D.1 Conventions, glossary, configuration

**Four operations that must never be conflated.**

| Term | Meaning | Performed by | Changes |
| --- | --- | --- | --- |
| Code deployment | New container images from a git commit reach the environment | CI/CD (`release` job, `make deploy-code TAG=...`) | Images only |
| Pipeline retraining | A training run on new data produces a candidate | `retraining` DAG | A registered model version |
| Model promotion | A version passes G1–G5 and receives alias `candidate`; after G7 it becomes `champion` | `train_and_register` and `deploy_candidate` DAGs | MLflow aliases |
| Model replacement | Traffic moves between slots | `deploy_candidate` via gateway | Gateway active slot |

**Glossary.** *Champion*: version serving live. *Candidate*: version approved by G1–G5 and not yet live. *Previous champion*: rollback target. *Slot*: `blue` or `green` container; *active* receives traffic, *idle* does not. *Window*: the last `window_rows` logged predictions. *Labelled row*: a logged prediction with a row in `labels`. *`data_version_id`*: SHA-256 of the data manifest (D.2). *Eval*, *test*, *stream*: partitions from N-06.

**Configuration is centralized** in `configs/`; code contains no thresholds. Files are validated at load by pydantic models in `src/fairxai/config.py`; unknown keys are an error. A profile (`FAIRXAI_PROFILE=demo|test`, default `base`) overlays `configs/profiles/<name>.yaml`.

| File | Owns |
| --- | --- |
| `base.yaml` | seed (42), paths, MLflow experiment `fairxai`, registered model name `fairxai-credit-default`, window sizes |
| `data.yaml` | column contract, ranges, categories, aliases, outlier rule (D.2) |
| `training.yaml` | model list, search spaces, CV folds, trials, `imbalance_strategy`, `n_jobs`, threshold policy |
| `fairness.yaml` | `favorable_label`, `age_bands`, tolerances, min sizes, warn margins |
| `gates.yaml` | PR-AUC floor and margin, G6 and G7 parameters, auto\_deploy (default true), `approval.model_promotion` |
| `monitoring.yaml` | window, tests, thresholds, labelled-row minimums |
| `retraining.yaml` | persistence N, cooldown, cap, minimum new rows, `new_data_weight`, holdout fraction |
| `serving.yaml` | `top_k` default, request limits, latency targets |
| `simulator.yaml` | scenarios, seeds, magnitudes, label delay |

**Environment variables** (documented in a committed `.env.example`; real `.env` is git-ignored): `POSTGRES_USER`, `POSTGRES_PASSWORD`, `FAIRXAI_DB_URL`, `MINIO_ROOT_USER`, `MINIO_ROOT_PASSWORD`, `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `MLFLOW_S3_ENDPOINT_URL`, `MLFLOW_TRACKING_URI`, `FAIRXAI_PROFILE`, `API_KEYS` (`client:sha256hex,...`), `ADMIN_API_KEY_SHA256`, `ADMIN_API_KEY` and `SIM_API_KEY` (plaintext, used only by Airflow, CLIs and simulator), `GATEWAY_URL`, `SLOT_BLUE_URL`, `SLOT_GREEN_URL`, `IMAGE_TAG`, `EXPECTED_MODEL_VERSION`, `NOTIFY_WEBHOOK_URL`, `NOTIFY_WEBHOOK_STYLE`, `GRAFANA_ADMIN_PASSWORD`, `AIRFLOW__CORE__FERNET_KEY`, Airflow admin user and password. Every variable has an example value that is obviously fake.

### D.2 Data contract, validation, ingestion, lineage

**Source.** UCI "Default of Credit Card Clients" (Taiwan, 2005), public benchmark data. The team places its copy at `data/raw/credit_default.csv`; `scripts/prepare_raw.py` accepts either the Kaggle-style CSV (one header row) or the UCI `.xls` (two header rows) and writes the canonical CSV below. Then `dvc add data/raw/credit_default.csv`, `dvc push` (remote: MinIO bucket `dvc`).

**Canonical columns** (from public documentation; *unverified against the team's file*; `scripts/profile_data.py` prints the real values in Phase 2 and its output must be reviewed before Phase 3):

| Column | Type | Valid range (hard check) | Notes |
| --- | --- | --- | --- |
| `ID` | int | unique | dropped before modelling |
| `LIMIT_BAL` | number | 1 to 2,000,000 | credit limit |
| `SEX` | int | {1, 2} | 1 male, 2 female; protected |
| `EDUCATION` | int | 0 to 6 | documented 1–4; 0, 5, 6 occur in public data and are mapped to 4 |
| `MARRIAGE` | int | 0 to 3 | 0 occurs and is mapped to 3 |
| `AGE` | int | 18 to 100 | protected; public data spans about 21–79 |
| `PAY_0`, `PAY_2` … `PAY_6` | int | −2 to 9 | repayment status; the public file has no `PAY_1` |
| `BILL_AMT1` … `BILL_AMT6` | number | −1,000,000 to 2,000,000 | may be negative |
| `PAY_AMT1` … `PAY_AMT6` | number | ≥ 0 |  |
| `default` | int | {0, 1} | target, 1 = default next month |

Aliases in `data.yaml` rename `default payment next month`, `default.payment.next.month`, `Y` to `default`, `PAY_1` to `PAY_0`, and `X1`…`X23` to the names above in file order. After renaming, the column set must match exactly or loading fails.

**Preprocessing (documented defaults; equivalence with Stage 1 is unverified, N-02).** Drop `ID`; no other cleaning in `prepare.py` (it does not remap categories; if it ever does, the remap must be an idempotent no-op). The **first step of the saved sklearn Pipeline** remaps EDUCATION {0,5,6}→ 4 and MARRIAGE 0 → 3, so training, serving and drift comparison share one remap. Scaling and encoding also live **inside the sklearn Pipeline** that is saved with the model, so training and serving use one artifact: one-hot for `EDUCATION` and `MARRIAGE` with `handle_unknown="error"`, `SEX` as 0/1, `PAY_*` numeric, `StandardScaler` for LR, KNN, SVM, passthrough for tree models and Naive Bayes. `features.include_protected: true` (SEX and AGE are model inputs; DEFAULTED, must be checked against Stage 1 and stated in the model card).

**Great Expectations suites** (`pipelines/expectations/`, wrapped by `fairxai.data.validate`):

- `raw_taiwan`: exact column set; exactly 30,000 rows; `ID` unique; every range above; no nulls; default rate between 0.15 and 0.30 as a warning only.
- `batch`: same checks except row count and `ID`; at least 100 rows; nulls forbidden; the target column is required only for labelled batches.
- Outlier rule (continuous columns `LIMIT_BAL`, `BILL_AMT*`, `PAY_AMT*`): a value is flagged outside `[Q1−3·IQR, Q3+3·IQR]` using train-partition statistics in `data/processed/reference_stats.json`. With `f_ref` the flagged fraction in the train partition, a batch **warns** when its fraction exceeds `max(2·f_ref, f_ref+0.05)` and **fails** above `f_ref+0.20`.
- Hard-check failure ⇒ batch `REJECTED`. Warnings are recorded and never block.
- Every validation writes one row per suite to `monitoring_results` (`kind='data_quality'`), giving quality over time.

**DVC stages** (`dvc.yaml`): `prepare` (raw → `data/processed/clean.csv`, `reference_stats.json`) and `split` (clean → `train.csv`, `eval.csv`, `test.csv`, `data/stream/batch_0001.csv` … `batch_0009.csv`; stratified on target x SEX, seed 42, stream rows kept in shuffled-once file order). `make data` runs `dvc repro` and `dvc push`. Airflow never runs `dvc repro`; its first task verifies that file hashes match `dvc.lock`.

**Ingestion** (`ingest_batch` DAG, parameter `path`):

1. Copy to `data/ingest/incoming/<batch_id>.csv`; compute SHA-256; if that hash is already `ACCEPTED`, stop (idempotent).
2. Insert `data_batches` row `INCOMING`; run suite `batch`.
3. Pass: move to `data/ingest/accepted/`, status `ACCEPTED`. Fail: move to `data/ingest/rejected/`, write `data/ingest/reports/<batch_id>.json`, status `REJECTED`, `notify(error)`, task fails. A rejected batch can never enter training.

**Lineage manifest.** `data_version_id = sha256(canonical_json({ "base": md5 of data/processed/train.csv from dvc.lock, "batches": sorted [ [batch_id, sha256], ... ] of the ACCEPTED batches used }))`. Training takes the batch list explicitly, so any run can be recreated from `git commit + data_version_id`.

### D.3 Training, tuning, reproducibility

**Contract.** `python -m fairxai.train [--batches id1,id2] [--profile base|demo|test]` (module `src/fairxai/models/train.py`). Input: `data/processed/train.csv` plus optionally ACCEPTED batches, `data/processed/eval.csv`, `test.csv`, configs. Output (stdout JSON and MLflow): `run_id`, `model_version`, `data_version_id`, `gate_decision`. Exit code 0 for APPROVED, REJECTED and INDETERMINATE (they are valid outcomes); non-zero only for errors. Idempotent for the same inputs and seed.

**Steps (in order, all deterministic).**

1. Load training frame; run gate **G1** (suite `batch` plus no REJECTED batch); compute `data_version_id`.
2. **Baselines.** The seven models with default parameters from `training.yaml`. Class imbalance: `class_weight="balanced"` for LR, Decision Tree, Random Forest, SVM; `scale_pos_weight = n_negative / n_positive` for XGBoost; none for KNN and Naive Bayes (N-21). Rank by mean PR-AUC (`average_precision_score`, default class as positive) over `StratifiedKFold(5, shuffle=True, random_state=42)` on train. Tie-break: model name ascending.
3. **Reported baseline table** (not used for selection): accuracy, precision, recall, F1, confusion matrix and PR-AUC of each baseline fitted on train and scored on eval at the N-07 threshold (the out-of-fold probabilities from step 2 are reused to choose each baseline's threshold). Logged to MLflow as `baselines.csv`.
4. **Shortlist** the top 3 (D21). **Tune** each with Optuna (`TPESampler(seed=42)`, 25 trials, 3-fold CV PR-AUC, 900 s limit, in-memory storage); every trial is a nested MLflow run. Search spaces are fixed in `training.yaml`: LR `C` log 1e-3–1e2; Decision Tree depth 2–15, leaf 1–50; Random Forest trees 100–400, depth 3–20, leaf 1–20; XGBoost trees 100–500, depth 2–8, learning rate log 0.01–0.3, subsample and colsample 0.6–1, min child weight 1–10 (`tree_method="hist"`); KNN neighbours odd 5–101 and weights; Naive Bayes `var_smoothing` log 1e-10–1e-6; SVM `C` and `gamma` log (R-3).
5. **Champion selection.** Order tuned models by CV PR-AUC. For each in order: fit on full train; choose the decision threshold from 5-fold out-of-fold train probabilities (grid 0.05–0.95 step 0.01, maximise F1, tie → closest to 0.5; N-07); score eval; if alias champion exists, score that champion freshly on the same eval set; compute fairness (D.4); run `decide()` (D.4). The first model whose decision is `APPROVED` becomes the candidate. If none is approved, the top-ranked model is registered with its decision and the run ends without alias `candidate`.
6. **Test report.** Once, for the selected model: performance and fairness on the test partition, logged with the `test_` prefix. Never read by `decide()`.
7. **Global SHAP summary** on a 1,000-row seeded train sample: `shap_importance.json` (mean absolute value per feature) and `shap_summary.png`.
8. **Log and register.** MLflow run holds: parameters, config snapshot (`config_snapshot.json`), `git_commit`, `git_dirty`, `data_version_id`, `dvc_train_md5`, Optuna summary, baselines, fairness and gate reports, the model (`mlflow.sklearn`, signature and input example). Register version under `fairxai-credit-default` with tags `gate_decision`, `decision_threshold`, `data_version_id`, `git_commit`, `run_id`. After registration the registered URI is loaded once and must give identical probabilities on the smoke set (G5). Alias `candidate` is set **only** for `APPROVED` and only after that check passes; on a mismatch G5 is recorded as FAIL, the `gate_decision` tag is rewritten to `REJECTED` and no alias is set.

Version tags also include `eval_pr_auc` (eval-set PR-AUC of the selected model), which D.7 and D.8 use as the champion's performance reference.

**Failure behaviour.** Any exception marks the MLflow run `FAILED`, leaves the registry untouched, calls `notify(error)` and exits non-zero. Airflow: training task 0 retries (deterministic); MLflow/network tasks 2 retries, 60 s apart; task timeout 60 min.

**Reproducibility.** Seed 42 everywhere (`random`, `numpy`, sklearn `random_state`, XGBoost `seed`, Optuna), `PYTHONHASHSEED=0`, fixed `n_jobs`. Test: two runs under `profile=test` (synthetic data, 3 trials) must select the same model with the same parameters and metrics within 1e-6 (N-27). CI never uses the real dataset: `tests/fixtures/make_synthetic.py` generates a seeded, schema-identical **synthetic** frame, always named as such.

**Tests.** Unit: split disjointness and stratification; threshold selection uses only train; imbalance handling per model; ranking tie-break; SMOTE (experiment only) never touches eval, test or stream; selection loop picks first APPROVED; EDUCATION=5 and MARRIAGE=0 give the same output as 4 and 3, and the one-hot encoder raises on an unseen category. Acceptance: Phase 3 and Phase 5 exit criteria in Section F.

### D.4 Fairness metrics and the promotion gate

**Definitions.** Let Ŷ be the predicted class (1 = default, using the model's decision threshold), Y the true class, G a group, and *f* the favorable label (`favorable_label`, default 0 = predicted not to default).

```latex
FR_g = P(\hat{Y}=f \mid G=g) \qquad TPR^{f}_g = P(\hat{Y}=f \mid Y=f,\, G=g)

EO = \max_g TPR^{f}_g - \min_g TPR^{f}_g \qquad DI = \frac{\min_g FR_g}{\max_g FR_g} \qquad DP = \max_g FR_g - \min_g FR_g

ECE_g = \sum_{b=1}^{10} \frac{n_{g,b}}{n_g}\,\bigl|\overline{y}_{g,b} - \overline{p}_{g,b}\bigr| \qquad CAL = \max_g ECE_g - \min_g ECE_g
```

The ECE bins are 10 equal-width bins on the predicted default probability, with *y* = 1 for default. Group sets: `SEX` {1, 2}; `age_band` with bands `[18,29]`, `[30,39]`, `[40,100]` labelled `18-29`, `30-39`, `40+` (public data starts at 21); `SEX x age_band` (6 cells) for the intersectional gate. MARRIAGE and EDUCATION get the same report tables and never gate.

**Inputs.** Arrays `y_true`, `y_prob`, the threshold, a DataFrame with `SEX`, `AGE` (and `MARRIAGE`, `EDUCATION`). Implementation: Fairlearn `MetricFrame` for per-group rates; calibration is about 15 lines of tested custom code in `fairness/calibration.py`. Rows with a missing group value or label are dropped and counted in `n_dropped`.

**Reliability rules.** A group is *eligible* for a metric when it has at least 100 rows **and** at least 50 rows in that metric's denominator (all rows for FR and calibration, rows with Y = f for TPR). Ineligible groups are excluded and listed. A hard-gated metric with fewer than 2 eligible groups, or with `max FR = 0`, is `INDETERMINATE`. All values are rounded to 6 decimals before comparison; equality with a threshold passes.

| Check | Rule (N-05) | Role |
| --- | --- | --- |
| `eo_sex`, `eo_age` | EO ≤ 0.10 | Hard |
| `di_sex`, `di_age` | DI ≥ 0.80 | Hard |
| `eo_inter` | EO over SEX x age\_band cells ≤ 0.15 | Hard |
| `di_inter` | DI over the same cells ≥ 0.80 | Hard |
| `dp_*`, `cal_*` | WARN when DP > 0.10 or CAL > 0.05 **and** (no champion or worse than champion by > 0.02) | Advisory |

**Report.** `fairness_report.json` (artifact) with keys `schema_version`, `favorable_label`, `n_rows`, `n_dropped`, `attributes.{sex,age_band,marriage,education}.groups.<g>.{n,n_denominator,favorable_rate,tpr,ece,eligible}`, `attributes.*.metrics`, `intersectional`, `gates`, `warnings`; also logged as MLflow metrics `fair_eo_sex`, `fair_di_sex`, …; plus `fairness_groups.csv`.

**Gates and the decision** (`registry/promotion.py::decide(candidate, champion_or_None, policy) -> GateDecision`). Each gate returns `PASS`, `FAIL`, `INDETERMINATE` (G4 returns `PASS` or `WARN`).

| Gate | Rule | Evaluated by |
| --- | --- | --- |
| G1 | Training data passed suite `batch` and has no REJECTED batch | `train` |
| G2 | No champion: PR-AUC ≥ 0.45. Champion exists: candidate PR-AUC ≥ champion PR-AUC + 0.01 on the **same** evaluation set. `INDETERMINATE` if the set has < 100 rows, one class, or a metric is NaN | `train` |
| G3 | All hard checks pass; any FAIL ⇒ FAIL; any INDETERMINATE and no FAIL ⇒ INDETERMINATE | `train` |
| G4 | Advisory rules; WARN is recorded and never blocks | `train` |
| G5 | Before registration, the in-memory pipeline is round-tripped through `mlflow.sklearn.save_model` to a temporary local path and loaded from there; it predicts finite probabilities in \[0,1\] on the fixed synthetic smoke set (`tests/fixtures/smoke_set.csv`); feature names and order equal the serving schema. After D.3 step 8 the registered URI is loaded once more and must give identical probabilities on the smoke set | `train` |
| G6 | Live-slot verification (N-13) | `deploy` |
| G7 | Post-flip observation (N-13) | `deploy` |
| G8 | Approval when `approval.model_promotion: true` | `deploy` |

**Aggregation.** Any mandatory gate FAIL ⇒ `REJECTED`. Otherwise any INDETERMINATE ⇒ `INDETERMINATE`. Otherwise `APPROVED`. `REJECTED` and `INDETERMINATE` both keep the champion, are logged and notified, and differ only in the message: INDETERMINATE means "cannot tell, investigate". Output JSON: `{decision, gates:{G1:{status,detail}...}, reasons:[...], warnings:[...], inputs_digest}`.

**Limitations (go into the model card).** Passing these thresholds does not make a model fair. The thresholds are conventional, not legal standards. Demographic parity, equal opportunity and calibration cannot in general all hold when base rates differ, which is why only the first two plus the intersectional check gate. Point estimates on 4,500 evaluation rows are noisy, especially for intersectional cells (R-2); X-14 adds intervals. If no candidate passes, nothing is promoted (R-1).

**Tests** (`tests/unit/fairness/`, `tests/unit/registry/`): hand-computed fixtures for every formula; boundary values exactly at 0.10, 0.15 and 0.80; a group of 99 rows; a group with 49 qualified rows; one eligible group; all-same-class group; `max FR = 0`; NaN probabilities; flipping `favorable_label` changes results as computed by hand; each gate returning each status; aggregation truth table including FAIL plus INDETERMINATE ⇒ `REJECTED`; a candidate that beats the champion on PR-AUC but fails G3 is `REJECTED`.

### D.5 Registry and lineage

**Registry module** `src/fairxai/registry/`. It is the only code that calls MLflow alias APIs (the exact function names must be checked against the pinned MLflow, \[VERIFY\]).

| Function | Effect | Failure behaviour |
| --- | --- | --- |
| `register_version(run_id, tags)` | Creates a version of `fairxai-credit-default` with the tags from D.3 | Raises; run stays unregistered |
| `set_candidate(v)` | Alias `candidate` → v. Allowed only if the version's `gate_decision` tag is `APPROVED` | Raises `NotApproved` |
| `promote_to_champion(v)` | After G7, and only if the approval guard below allows it: `previous_champion` ← current `champion`; `champion` ← v; remove `candidate` | Raises; aliases unchanged (single transaction is not available, so the function sets `previous_champion` first and verifies all three aliases afterwards, raising `RegistryInconsistent` if they differ) |
| `rollback_target()` | Version behind `previous_champion`, or `None` | Returns `None` |

**Approval guard.** `promote_to_champion(v)` and `deploy(v)` raise `NotApproved` unless the version's `gate_decision` tag is `APPROVED`, or the version previously held alias `champion` or `previous_champion` (so rollback stays possible). The tag `FIXTURE_FORCED` (D.9) is accepted only under profiles `demo` and `test`.

Every transition writes one row to `events` (`source='registry'`, `details = {actor, version, from, to, reason, gate_results}`) and a version tag `last_transition`. Actor format: `airflow:<dag_id>:<run_id>` or `cli:<os user>`.

**Lineage.** Every deployed version resolves to: MLflow `run_id`; `git_commit` and `git_dirty` (release deployments require `git_dirty=false`); `dvc_train_md5`; `data_version_id`; `config_snapshot.json`; Optuna summary artifact; `image_tag` (the serving image running the model, from env `IMAGE_TAG`); `deployment_id` (row in `deployments`). `fairxai.registry.lineage.resolve(version)` returns all of them and raises if any is null. A test deploys a fixture model in the integration stack and asserts every field is present and equals the recorded value.

### D.6 Serving and gateway API contracts

**General rules.** JSON over HTTP, UTF-8, request bodies ≤ 64 KB (413 otherwise). Authentication: header `X-API-Key`, compared as SHA-256 hash in constant time against `API_KEYS`; admin endpoints require `ADMIN_API_KEY_SHA256`. `/healthz`, `/readyz`, `/metrics` are unauthenticated and the compose file publishes ports on `127.0.0.1` only. Keys are never logged or returned. Unknown JSON fields are rejected. Every response carries `X-Request-ID`. Error body for every non-2xx:

```json
{"error": {"code": "validation_error", "message": "human readable", "request_id": "uuid", "details": [{"field": "features.AGE", "problem": "must be between 18 and 100"}]}}
```

| Status | `code` | When |
| --- | --- | --- |
| 401 | `unauthorized` | Missing or wrong key |
| 403 | `forbidden` | Valid client key on an admin route |
| 404 | `not_found` | Unknown route, model version or request id |
| 409 | `slot_not_ready` | Gateway flip to a slot whose `/readyz` is not 200 |
| 413 | `payload_too_large` | Body > 64 KB |
| 422 | `validation_error` | Any schema or range violation (the FastAPI default format is replaced by a handler) |
| 500 | `internal_error` | Unexpected failure; no stack trace in the body |
| 502, 504 | `upstream_unavailable`, `upstream_timeout` | Gateway cannot reach or wait for the slot |
| 503 | `model_not_ready` | No model loaded in the slot |

**Slot endpoints** (ports 8001 blue, 8002 green; the gateway exposes `/v1/*` on 8080).

| Method and path | Request | Success response | Notes |
| --- | --- | --- | --- |
| `POST /v1/predict` | `{"request_id"?: uuid, "features": {23 fields}}` | 200 `{request_id, model_name, model_version, prediction (0 or 1), probability_default, decision_threshold}` | All 23 features required, no nulls, ranges from `data.yaml`; `prediction = probability_default >= decision_threshold`. Logged to `predictions` with latency. Invalid requests are **not** logged as predictions. A repeated `request_id` is ignored by the log (`ON CONFLICT DO NOTHING`). A request carrying header X-Probe: true is scored and returned but never logged; deployment probes use it. If the log write fails the prediction is still returned with header `X-Logged: false`, and `fairxai_log_failures_total` increases |
| `POST /v1/explain?top_k=10` | same body | 200 `{request_id, model_version, method, output_space, base_value, probability_default, contributions: [{feature, value, shap_value}]}` sorted by absolute `shap_value`, `top_k` 1–23 | Uses the loaded model and N-23. `method` is `tree_shap`, `linear_shap` or `kernel_shap`; `output_space` is `log_odds` or `probability`, stated as the explainer returns it. Not logged as a prediction. Never falls back to another method silently: an explainer failure returns 500 `internal_error` |
| `POST /v1/feedback` | `{"labels": [{"request_id": uuid, "default": 0 or 1}]}` (1–1000) | 200 `{accepted, duplicates, unknown_request_ids: []}` | Writes `labels`; a conflicting repeat label is a duplicate, not an overwrite |
| `GET /v1/model` | none | 200 `{model_name, model_version, run_id, loaded_at, decision_threshold, git_commit, data_version_id, slot}` |  |
| `GET /healthz` | none | 200 `{"status":"ok"}` | Process alive |
| `GET /readyz` | none | 200 `{"ready":true,"model_version":7}` or 503 `{"ready":false,"reason":"..."}` | 200 only when a model is loaded and `model_version` equals the version last requested by `/admin/load` or `EXPECTED_MODEL_VERSION` |
| `GET /metrics` | none | Prometheus text | See below |
| `POST /admin/load` | `{"model_version": 7}` (admin key) | 200 `{loaded:true, model_version:7, previous_version:6}` | Loads from `models:/fairxai-credit-default/7`, builds the explainer, then swaps atomically. On any failure the slot **keeps its previous model**, returns 404 (unknown version) or 500, and increments `fairxai_model_load_failures_total` |

**Slot startup.** On startup a slot loads `EXPECTED_MODEL_VERSION` if set, otherwise stays 503 `model_not_ready`.

**Gateway endpoints.** `/v1/*` forwards to the active slot (connect timeout 1 s, read timeout 5 s for predict and feedback, 15 s for explain; no automatic retries). `GET /healthz`; `GET /readyz` (200 only if the active slot is ready); `GET /metrics`; `GET /admin/state` → `{active, slots:{blue:{url,ready,model_version}, green:{...}}, updated_at}`; `POST /admin/active {"slot":"blue|green","reason":"..."}` (admin key) → 200 `{active, previous, ts}` or 409 `slot_not_ready`. The active slot is stored in the Postgres table `gateway_state` (single row, default `blue`) and survives restarts.

**Metrics.** Slots: `fairxai_requests_total{endpoint,status,slot,model_version}`, `fairxai_request_latency_seconds{endpoint,slot}` (histogram), `fairxai_predictions_total{prediction,model_version}`, `fairxai_model_loaded{model_version}`, `fairxai_model_load_failures_total`, `fairxai_log_failures_total`. Gateway: `gateway_active_slot{slot}`, `gateway_flips_total`, `gateway_upstream_errors_total{slot,code}`, `gateway_request_latency_seconds`. Prometheus alert rules (`infra/prometheus/alerts.yml`): 5xx rate > 5 % for 2 min; p95 predict latency > 0.5 s for 5 min; slot `up == 0` for 1 min; active slot with `fairxai_model_loaded == 0`; `fairxai_log_failures_total` increasing.

**Example** (illustrative request; the response values are placeholders, not measured output):

```json
POST /v1/predict   X-API-Key: <key>
{"features": {"LIMIT_BAL": 20000, "SEX": 2, "EDUCATION": 2, "MARRIAGE": 1, "AGE": 24,
  "PAY_0": 2, "PAY_2": 2, "PAY_3": -1, "PAY_4": -1, "PAY_5": -2, "PAY_6": -2,
  "BILL_AMT1": 3913, "BILL_AMT2": 3102, "BILL_AMT3": 689, "BILL_AMT4": 0, "BILL_AMT5": 0, "BILL_AMT6": 0,
  "PAY_AMT1": 0, "PAY_AMT2": 689, "PAY_AMT3": 0, "PAY_AMT4": 0, "PAY_AMT5": 0, "PAY_AMT6": 0}}

200 {"request_id": "3f0c...", "model_name": "fairxai-credit-default", "model_version": 1,
     "prediction": 1, "probability_default": 0.0, "decision_threshold": 0.0}
```

**Tests** (`tests/api/`): schema accept and reject for each field boundary; every status code above; auth with right, wrong, missing and admin keys; keys absent from logs and bodies; `/readyz` 503 before load, after failed load and on version mismatch; a failed `/admin/load` leaves the previous model serving; explain additivity (base + sum of contributions equals the model output in `output_space` within 1e-4) and `method` matches the model family; feedback duplicates and unknown ids; log-failure header; metric names present after traffic; gateway refuses a flip to a not-ready slot and survives a restart with the stored slot.

### D.7 Monitoring and drift

Statuses written to `monitoring_results.status`: `OK`, `WARN`, `ALERT`, `NA` (not computable; always with the reason and `n_rows`, `n_labelled`). **Feature drift is a signal to investigate; it is never evidence of performance degradation.** Only labelled data can confirm degradation. The monitoring DAG is triggered once per window (by the simulator or `make monitor`); an optional schedule is `monitoring.schedule`, default `None`.

| Signal | Data source | Method | Reference | Evaluation window | Threshold | Min. data | Action |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Data quality | `predictions.features` | Suite `batch` (D.2) | Contract and train statistics | Last 500 rows | Hard failure ⇒ ALERT; outlier rule ⇒ WARN | 300 rows else NA | Event and notification; never retrains by itself |
| Feature drift | same | Evidently adapter: per feature KS (numeric) or chi-square (categorical), p < 0.05 means drifted. The logged features and the train reference both get the D.2 EDUCATION and MARRIAGE remap before comparison | Train partition | Last 500 rows | Dataset drift when ≥ 30 % of features drifted ⇒ ALERT; any single feature ⇒ WARN | 300 rows else NA | Tier 1: record. Counts toward persistence N = 3 consecutive windows |
| Prediction drift | `predictions` | KS on `probability_default`; absolute change in predicted-default rate | Eval partition scored by the champion | Last 500 rows | KS p < 0.05 or rate change ≥ 0.10 ⇒ WARN | 300 rows else NA | Record only |
| Model performance | `predictions` joined to `labels` | PR-AUC, F1, precision, recall | Champion's eval PR-AUC (stored at promotion) | Most recent 500 labelled rows | PR-AUC ≤ reference − 0.05 ⇒ ALERT; counts toward 2 consecutive windows | 300 labelled rows else NA | Confirmed degradation after 2 windows ⇒ retrain request |
| Live fairness | same labelled rows | D.4 functions with the same eligibility rules | N-05 thresholds | same labelled window | Hard-check violation ⇒ ALERT; advisory ⇒ WARN | 300 labelled rows else NA | Notification; never retrains by itself |
| Service health | Prometheus | Rules in D.6 | n/a | Rule specific | Rule specific | n/a | Visible in Prometheus and Grafana; webhook only with profile `alerting` |
| Pipeline health | Airflow | `on_failure_callback` calls `notify(error)` | n/a | per run | any failed task | n/a | Event row and optional webhook |

**No labels.** If fewer than 300 labelled rows exist, performance and live fairness are `NA`, the report states how many rows were labelled, and nothing is inferred from drift. Drift persistence alone can still *request* retraining, but a request always needs at least 500 unused labelled rows (D.8), otherwise it is suppressed with reason `insufficient_labelled_data`.

**Persistence bookkeeping.** `monitoring_results` rows for drift and performance carry `window_end`; the trigger evaluator reads the latest N rows for the current `model_version`. A model change resets the counters. Windows with status `NA` neither extend nor reset a streak.

**Grafana.** Provisioned from `infra/grafana/` (datasources Prometheus and Postgres; no manual clicking). Dashboards: *Service* (latency, throughput, errors, active slot), *Model* (drift share, PR-AUC over time, labelled-row count), *Fairness* (each hard check against its threshold, per group rates), *Lifecycle* (champion version, deployments, rollbacks, retraining events).

**Tests.** Fixtures for every row of the table at, just below and just above each threshold; persistence streaks with an `NA` window in between; model change resets; adapter returns the same dataclass for a stub Evidently result (and for the scipy contingency); a drift-only fixture never produces a retrain request while one with 3 windows does.

### D.8 DAGs, retraining, deployment, rollback

DAGs contain orchestration only; each task calls a function or CLI in `src/fairxai/`. `catchup=False`, `schedule=None` unless stated.

| DAG | Trigger | Tasks | Limits |
| --- | --- | --- | --- |
| `ingest_batch` | param `path` | copy\_hash → validate → accept\_or\_reject → notify | `max_active_runs=4`, task timeout 10 min, 2 retries on I/O |
| `train_and_register` | manual or `make train` | verify\_data → train → branch(APPROVED: set\_candidate → trigger `deploy_candidate` if `auto_deploy` (default true); otherwise stop) | `max_active_runs=1`, timeout 60 min, 0 retries |
| `deploy_candidate` | param `version` (default alias `candidate`) | preflight → load\_idle → G6 → approval → flip → probe\_and\_observe (G7) → finalize\_or\_rollback | `max_active_runs=1`, timeout 30 min |
| `monitoring` | per window | collect → quality → drift → prediction\_drift → performance → fairness → write → evaluate\_triggers → maybe trigger `retraining` | `max_active_runs=1`, timeout 15 min |
| `retraining` | trigger request | acquire → throttle → export\_labelled → ingest → train → decide → trigger deploy → finalize | `max_active_runs=1`, timeout 90 min |
| `retention_purge` | `@daily` | delete rows older than 90 days from `predictions` and `labels` | 1 retry |
| `rollback` | manual, param `version` (default `previous_champion`) | same steps as `deploy_candidate` for that version | `max_active_runs=1` |

**Deployment sequence** (`src/fairxai/deploy/deploy.py::deploy(version)`; every step writes to `deployments` with statuses `STARTED`, `LOAD_FAILED`, `G6_FAILED`, `AWAITING_APPROVAL`, `FLIPPED`, `DEPLOYED`, `ROLLED_BACK`):

1. **Preflight.** Apply the approval guard of D.5 (`NotApproved` stops the deployment before any load). Read `GET /admin/state`. Idle slot = the one not active. Refuse (error) if the idle slot is not reachable.
2. **Load.** `POST idle/admin/load {version}`. Failure ⇒ `LOAD_FAILED`; the idle slot keeps its previous model; the gateway is untouched.
3. **G6.** Send 200 rows of `eval.csv` (header `X-Probe: true`, which makes the slot skip logging) directly to the idle slot: no errors, p95 ≤ 0.5 s. Failure ⇒ `G6_FAILED`; champion untouched.
4. **G8.** If `approval.model_promotion` is true, set `AWAITING_APPROVAL` and stop; `make approve VERSION=n` re-enters at step 5.
5. **Flip.** `POST /admin/active {slot: idle}`; the gateway refuses with 409 if the slot is not ready. Status `FLIPPED`.
6. **G7.** Send 100 probe requests through the gateway and observe at least 120 s (maximum 600 s). Pass when probe 5xx rate ≤ 1 %, probe p95 ≤ 0.5 s and, if real traffic exists, the Prometheus 5xx ratio over the window is ≤ 1 %. Inconclusive evidence (no responses) counts as failure.
7. **Finalize.** Pass: `promote_to_champion(version)`, status `DEPLOYED`, read the champion's eval PR-AUC from the version tag eval\_pr\_auc (written by train) as the performance reference. Fail: flip back to the previous slot, aliases unchanged, status `ROLLED_BACK`, `notify(error)`.

**First deployment.** If alias `champion` does not exist, there is no live model: load into the active slot (blue), run G6 and 20 probes, set `champion`, skip flip and observation, status `DEPLOYED`.

**Manual rollback.** `make rollback` runs the `rollback` DAG: the same sequence for `previous_champion`. A loaded model on the idle slot can also be flipped to by `POST /admin/active`. No step ever stops or unloads the champion's slot before G7 passes; a failed load never changes a slot.

**Retraining sequence** (`src/fairxai/retrain/retrain.py`). Table `retrain_events(id, trigger, status, data_version_id, candidate_version, started_at, finished_at, reason, gate_report)` with a partial unique index `ON retrain_events ((true)) WHERE status = 'RUNNING'`.

1. **Acquire.** Insert a `RUNNING` row. A unique-index violation ⇒ this request is recorded as `SUPPRESSED` (reason `concurrent_run`) and the task ends successfully. Before inserting, any RUNNING row older than the DAG timeout is first marked FAILED with reason stale, and its exported labels are released as in step 7. This makes repeated signals idempotent.
2. **Throttle** (`retraining.yaml`, demo profile overrides): `SUPPRESSED` with reason `cooldown` if the last non-suppressed event finished within 60 minutes; `cap` if 3 non-suppressed events started in the last 7 days. Applies to every trigger, including scheduled.
3. **Export labelled rows** not yet used (`labels.exported_at IS NULL`). In the same transaction as the export, set `labels.exported_at` and `labels.exported_event_id` on every exported row. Fewer than 500 ⇒ `SUPPRESSED` (`insufficient_labelled_data`); a scheduled trigger needs at least 1,000. Rows are split by `int(sha256(request_id)[:8], 16) mod 100 < 30` into `recent_eval.csv` (30 %) and a batch file (70 %); both live under `data/retrain/<event_id>/`.
4. **Ingest** the batch through the same validation as D.2. Rejected ⇒ event `FAILED` (reason `validation`), champion kept, alert.
5. **Train** with the same code and configuration, model family and tuned hyperparameters of the current champion (no re-tuning, no baselines), new rows weighted by `new_data_weight` (3). Evaluation set = `eval.csv` plus `recent_eval.csv`; the champion is scored on the identical set.
6. **Decide** with `decide()` (G1–G5). `APPROVED` ⇒ alias `candidate`, trigger `deploy_candidate`. Otherwise record the decision, keep the champion.
7. **Finalize** the event with its terminal status: `FAILED`, `REJECTED`, `INDETERMINATE`, `APPROVED` (awaiting or in deployment), `DEPLOYED`, `ROLLED_BACK`. If the event ends `FAILED`, reset `labels.exported_at` and `labels.exported_event_id` to NULL for its rows; for every other terminal status they stay set, so rows are consumed once. A crash leaves a `RUNNING` row, so a startup task in `retraining` marks any `RUNNING` row older than the DAG timeout as `FAILED` (reason `stale`).

**Notifications.** `notify(severity, source, message, details)` always inserts into `events` and logs; if `NOTIFY_WEBHOOK_URL` is set it posts `{"text": ...}` (`slack`) or `{"content": ...}` (`discord`), with a 5 s timeout and no retry; a webhook failure never fails the pipeline.

**Tests.** Concurrency (two simultaneous requests ⇒ one run, one `SUPPRESSED`); each throttle reason; stale `RUNNING` cleanup; insufficient labelled data; failed validation keeps the champion; rejected candidate keeps the champion serving; deploying a REJECTED version fails; label consumption (exported rows are not exported again; they are released only when the event ends `FAILED`); deployment fixtures for `LOAD_FAILED`, `G6_FAILED`, G7 failure with rollback, first deployment, approval on and off; manual rollback; a failed step never leaves the gateway pointing at a slot that was not ready.

### D.9 Simulator, faults and the deterministic demonstration

**Simulator** (`python -m fairxai.simulator ...`, profile `sim`). Replays `data/stream/batch_0001..0008.csv` (500 rows each, one batch = one monitoring window); `batch_0009.csv` is reserved for the ingestion demo. Sends rows to the gateway with `SIM_API_KEY`, 4 concurrent workers, at most 20 requests/s, `request_id` derived from `uuid5(seed, batch, row)` so reruns are idempotent. Labels return through `/v1/feedback` **one batch later** (`label_delay_batches: 1`), not by wall-clock time. All randomness uses seed 42. **Order for batch n:** (1) send predictions(n); (2) post labels(n−1) through `/v1/feedback`; (3) trigger the `monitoring` DAG for window n and wait for the run to finish (timeout 5 min). Faults alter the labels of the batches they are applied to, so labels(n), posted during batch n+1, are already altered.

**Faults** (`simulator.yaml`; starting magnitudes, final values are recorded after the tuning step below):

| Fault | Effect | Checked by |
| --- | --- | --- |
| `covariate_shift` | `LIMIT_BAL` × 0.4; `BILL_AMT1` × 1.5; `PAY_0`, `PAY_2`…`PAY_6` + 2 (clipped to 8). Eight of 23 features move, so dataset drift (≥ 30 %) is reachable. True labels untouched | Drift ALERT on each shifted window |
| `concept_shift` | For rows with `PAY_0 ≥ 1`, the label is set to 0 with probability 0.7 (seeded) | Champion PR-AUC drop ≥ 0.05 on the labelled window |
| `corrupted_batch` | Three variants, each a hard validation failure: column `AGE` removed; 10 % of `SEX` set to 3; 5 % of `LIMIT_BAL` null | Ingestion `REJECTED`, report written |

**Tuning step (Phase 12, once).** Increase the concept-shift probability in steps of 0.1 from 0.7 until the champion's PR-AUC on the labelled window falls by at least 0.08 (margin over the 0.05 threshold); record the final values and the measured drops in `docs/evidence/fault_tuning.md`. Never edit a threshold to make a scenario pass.

**Forced fixtures** (`src/fairxai/simulator/fault_models.py`; usable only when `FAIRXAI_PROFILE` is `demo` or `test`, refused under `base`, registered with tags `fixture=true` and `gate_decision=FIXTURE_FORCED`, except `biased`, which keeps the real `decide()` result `REJECTED` and is never deployed; `deploy()` accepts `FIXTURE_FORCED` only under those two profiles, D.5). They exercise gate and deployment logic deterministically; their results are never presented as organic model behaviour.

| Fixture | Construction | Expected outcome |
| --- | --- | --- |
| `good_copy` | Champion model re-registered as a new version | `deploy` ends `DEPLOYED` |
| `biased` | Champion plus a group-specific threshold shift for SEX = 2, increased in steps of 0.05 until `eo_sex` > 0.12 on eval (error if not reached in 10 steps) | `decide()` returns `REJECTED` (G3) |
| `broken_artifact` | Version whose model file is truncated | `/admin/load` fails; deployment `LOAD_FAILED`; same slot still serves |
| `slow` | Wrapper sleeping 0.8 s per prediction | `G6_FAILED` |
| `erroring` | Wrapper that works for 200 calls, then raises | Passes G6, fails G7, `ROLLED_BACK` |
| `fixture_champion` | Used only when the real gate decision in demo step 0 is not `APPROVED`. A model trained by `python -m fairxai.train --profile test` on seeded synthetic data (`tests/fixtures/make_synthetic.py`, seed 42, 3 trials), registered with `fixture=true` and `gate_decision=FIXTURE_FORCED`, then deployed through the first-deployment path | `deploy` ends `DEPLOYED`; it is the champion for steps 1–7 |

**Deterministic demonstration** (`make demo`, profile `demo`: cooldown 5 min, cap 3, `fairxai.clock.now()` offset advanced by the script instead of waiting). Every step asserts and prints PASS or FAIL, and the run writes `docs/evidence/demo_<UTC timestamp>.md` from the real outputs.

| Step | Action | Asserted outcome |
| --- | --- | --- |
| 0 | `make reset`, `make data`, `make train` | The real gate decision is recorded. If `APPROVED`, the real champion is `DEPLOYED`. If not, the evidence file records the decision and R-1, and the demo continues on a `fixture_champion` (`DEPLOYED`) |
| 1 | Ingest the three corrupted variants, then `batch_0009` | Three `REJECTED` with reports; one `ACCEPTED` |
| 2 | Replay batches 1–2 (no fault) | Drift `OK`; performance computed from batch 2 onward; `/v1/predict` and `/v1/explain` return 200 |
| 3 | Replay batches 3–5 with `covariate_shift` | Drift `ALERT` three windows in a row; one retrain request; event reaches a terminal status; either way the gateway still serves a registered version |
| 4 | Trigger monitoring again without advancing the clock | Retrain request `SUPPRESSED` (`cooldown`) |
| 5 | Fixtures `biased`, `broken_artifact`, `slow`, `erroring`, then `good_copy` (clock advanced) | `REJECTED`; `LOAD_FAILED`; `G6_FAILED`; `ROLLED_BACK`; `DEPLOYED`. After each failure `GET /v1/model` through the gateway returns the previous version and 20 predictions succeed |
| 6 | Replay batches 6–8 with `concept_shift` | Performance `ALERT` in window 7 and window 8 (two consecutive; window 6 uses the unaltered labels of batch 5 and is not asserted); one retrain request; terminal status recorded; the last approved model is still serving if the candidate failed any gate |
| 7 | Compare the registry alias `champion` with `GET /v1/model` | Equal |

**Two labelled outcomes.** Step 0 yields either a *real-model demo* (real gate `APPROVED`) or a *fixture-champion demo*. In the second case the evidence file marks every result after step 0 as "fixture champion; not a real-model result", and its last line states which outcome occurred.

Step 3 and step 6 assert that the pipeline completes and that the recorded decision equals what `decide()` computes from the logged reports; they do **not** pre-assert `APPROVED`, because an organic retrain may legitimately be rejected (R-4). The approved-promotion path is proven by step 5 and by the unit tests of `decide()`.

### D.10 CI/CD

Platform: GitHub Actions (D9). Workflow `.github/workflows/ci.yml` runs on every push and pull request; `release.yml` runs on tags `v*` and `workflow_dispatch`. No job may use `continue-on-error`; no gate may be skipped to obtain a green build.

| Job | Needs | Steps | Fails when |
| --- | --- | --- | --- |
| `lint-test` | none | install from lock files; `ruff check`; `ruff format --check`; `pytest tests/unit tests/data tests/model tests/api`; DAG import and config-schema tests; `promtool check rules` | any command exits non-zero |
| `security` | none | gitleaks over full history; `pip-audit` on every lock file | a secret or an unignored known vulnerability is found |
| `build` | `lint-test`, `security` | build images `serving`, `gateway`, `airflow`, `simulator` tagged `fairxai-<name>:<short sha>`; Trivy scan of each | fixable HIGH or CRITICAL finding outside the reviewed `.trivyignore`; build error |
| `integration` | `build` | start compose (`core` without Airflow webserver and scheduler, plus `docker-compose.ci.yml`); run `pytest tests/integration` on synthetic data: ingest, train, register, deploy, predict, explain, feedback, monitoring computation, retrain decision, rollback, failure recovery, lineage; always dump logs and `docker compose down -v` | any test fails or a service is unhealthy after 3 min |
| `e2e-full` (manual `workflow_dispatch` only) | `build` | full stack with Airflow, `make demo` under profile `test` | any assertion fails |
| `release` (`release.yml`, environment `production`) | all of the above on the same commit | require clean tree and `git_dirty=false`; retag images with the release tag; upload `images.tar` and `docs/evidence/` as workflow artifacts; optional GHCR push (X-11) | approval refused (required reviewer, \[VERIFY\] availability; fallback: `workflow_dispatch` with a typed confirmation input), or any upstream job failed |

A GitHub-hosted runner cannot reach a locally run stack, so **"production" is an approved, scanned, tagged release artifact**. It reaches the local machine by `make deploy-code TAG=<tag>` (`scripts/deploy_code.sh`): remember the current tag in `.deployed_tag`, `docker load` or pull, set `EXPECTED_MODEL_VERSION` for the active slot from alias `champion` and for the idle slot from `previous_champion` when it exists, `docker compose up -d --wait` for the new tag, run `scripts/smoke.sh` (health of every service, `GET /v1/model` through the gateway returns the champion version, one prediction). If the smoke test fails the script restores the previous tag and exits non-zero. The deployment is reported successful only when the smoke test passed. **Zero-downtime is not claimed**: recreating containers interrupts service. Only *model replacement* keeps traffic on the champion; the integration test sends requests continuously during a flip and records the number of non-2xx responses, which must be 0 in that test.

**Image tags.** `fairxai-<name>:<short sha>`; releases add `<tag>`; the running tag is exposed as `IMAGE_TAG` and recorded on deployments. **Dependency policy.** Lock files only; Dependabot or manual upgrades go through a pull request with the full CI run. **Pre-commit** hooks: gitleaks, ruff, `check-added-large-files`.

### D.11 Security, resources, operations

**Security (proportionate).** No secret in git or images; `.env` git-ignored; `.env.example` lists every variable with a fake value; gitleaks in CI and pre-commit. Client API keys stored only as SHA-256 hashes; plaintext keys exist only in `.env` for the simulator and Airflow. All published ports bind to `127.0.0.1`. Containers run as non-root where the base image allows. Serving, gateway and Airflow have no Docker socket. Logs never contain keys or full request bodies at INFO level. Credit data in this project is a public benchmark; the 90-day purge (N-16) and absence of direct identifiers are the privacy controls. Enterprise controls (audit log, protected-attribute isolation, Vault) are X-07 and X-13.

**Hardware and resources.** The host specification is unknown (N-28). Provisional budget: 4 cores, 16 GB RAM, 20 GB disk. Estimated idle memory (not measured): Airflow 1.5–2 GB, serving 2 x 0.4 GB, MLflow 0.4 GB, others about 0.1–0.2 GB each, so roughly 4–5 GB for the stack plus training headroom. Estimated first start after images are built: 1–3 minutes. Training time is unknown; the 900 s per-model Optuna limit bounds tuning (3 models, so at most about 45 minutes plus baselines). **Phase 0 and Phase 11 must record real numbers in `docs/resource-report.md`** (peak memory per container from `docker stats`, start time, training duration, p95 latency at 5 requests/s); no compatibility claim is made until then. Compose sets memory limits of 1 GB for serving slots and 2.5 GB for Airflow containers as starting values.

**Operating procedure** (all through the `Makefile`; every target is idempotent):

| Command | Effect |
| --- | --- |
| `make env` | Copy `.env.example` to `.env` if absent and generate local keys and hashes |
| `make up` / `make up-lite` | Start `core` / start `core` without Airflow (development without orchestration) |
| `make down`, `make ps`, `make health`, `make logs SVC=<name>` | Stop, status, health of every service, tail logs |
| `make reset` | `down -v`, clear `data/ingest`, `data/retrain`, `.deployed_tag`; raw and DVC-tracked data are kept |
| `make data` | `dvc repro` and `dvc push` |
| `make train`, `make deploy`, `make approve VERSION=n`, `make rollback`, `make monitor` | Trigger the corresponding DAG or CLI and wait |
| `make demo` | Run D.9 end to end |
| `make test`, `make lint` | Local equivalents of CI jobs; make test-integration runs tests/integration from a clean state |

A one-shot `init` service creates the MinIO buckets, the three Postgres databases and the `fairxai` schema from `infra/postgres/schema.sql`, then exits. Troubleshooting (`docs/runbook.md`): a port in use → change the published port in `.env`; MLflow cannot write artifacts → check `AWS_*` and `MLFLOW_S3_ENDPOINT_URL`; slot `readyz` 503 → `GET /admin/state`, then `POST /admin/load`; gateway 409 → target slot not ready, load a model first; out of memory → `make up-lite` and run DAG logic through CLIs; Airflow import errors → `airflow dags list-import-errors` in the scheduler container.

## E. Repository implementation blueprint

The top-level folders follow D20. Nothing below exists without a purpose listed in the tables. File names inside `src/fairxai/` are binding because tests and DAGs import them.

```text
fairxai-mlops/
  README.md  Makefile  pyproject.toml  .env.example  .gitignore  .gitleaks.toml
  .pre-commit-config.yaml  .trivyignore  dvc.yaml  dvc.lock  .dvc/config
  requirements/        app|airflow|serving|dev  (.in and .lock)
  configs/             base data training fairness gates monitoring retraining serving simulator (.yaml)
    profiles/          demo.yaml  test.yaml
  data/
    raw/               credit_default.csv.dvc
    processed/         clean.csv train.csv eval.csv test.csv reference_stats.json   (DVC outputs)
    stream/            batch_0001.csv ... batch_0009.csv                            (DVC outputs)
    ingest/            incoming/ accepted/ rejected/ reports/                       (generated)
    retrain/           <event_id>/ batch.csv recent_eval.csv                        (generated)
  src/fairxai/
    config.py clock.py db.py notify.py logs.py
    data/              load.py prepare.py split.py validate.py ingest.py
    features/          pipeline.py
    models/            zoo.py imbalance.py tune.py threshold.py evaluate.py train.py
    fairness/          groups.py metrics.py calibration.py report.py
    explain/           shap_global.py shap_local.py
    registry/          registry.py promotion.py lineage.py
    monitoring/        windows.py drift.py performance.py triggers.py run.py
    retrain/           retrain.py
    deploy/            deploy.py probes.py
    simulator/         __main__.py replay.py faults.py fault_models.py
  pipelines/
    dags/              ingest_batch.py train_and_register.py deploy_candidate.py monitoring.py
                       retraining.py retention_purge.py rollback.py
    expectations/      raw_taiwan.json batch.json
  serving/             Dockerfile  app/{main,schemas,auth,model_store,metrics,logging_db}.py
  gateway/             Dockerfile  app/main.py
  simulator/           Dockerfile
  infra/
    compose/           docker-compose.yml docker-compose.ci.yml
    airflow/           Dockerfile
    postgres/          schema.sql
    prometheus/        prometheus.yml alerts.yml
    grafana/           provisioning/{datasources,dashboards}/  dashboards/{service,model,fairness,lifecycle}.json
  scripts/             prepare_raw.py profile_data.py smoke.sh deploy_code.sh demo.sh check_tree.py
  experiments/         stage3_imbalance.py  cross_dataset/   (optional, outside production DAGs)
  tests/
    unit/ data/ model/ api/ integration/    fixtures/{make_synthetic.py, smoke_set.csv}
  docs/
    SPEC.md  decision_log.md  architecture.md  runbook.md  model_card.md  cloud-path.md
    resource-report.md  evidence/
  .github/workflows/   ci.yml  release.yml
```

| Path | Purpose | Owner of configuration | Source-controlled or generated |
| --- | --- | --- | --- |
| `configs/` | The only place for thresholds, tolerances, windows, policies, seeds | `fairxai.config` loads and validates; nothing else reads YAML | Source-controlled |
| `data/raw/*.dvc`, `dvc.yaml`, `dvc.lock` | Pointers and pipeline of versioned data | DVC | Source-controlled; the data itself is in the MinIO remote |
| `data/processed`, `data/stream` | Cleaned data and splits | `dvc repro` | Generated (DVC outputs, git-ignored) |
| `data/ingest`, `data/retrain` | Quarantine, accepted and rejected batches, retraining exports | `fairxai.data.ingest`, `fairxai.retrain` | Generated, git-ignored |
| `src/fairxai/` | All logic; unit-testable without Airflow or Docker | `pyproject.toml` | Source-controlled |
| `pipelines/dags/` | Orchestration only (D.8) | Airflow | Source-controlled |
| `pipelines/expectations/` | Great Expectations suites | `fairxai.data.validate` | Source-controlled |
| `serving/`, `gateway/` | Slot and gateway FastAPI apps; both import `fairxai` for config, DB and metrics helpers | `serving.yaml`, env | Source-controlled |
| `simulator/` | Dockerfile only; code is `src/fairxai/simulator/` | `simulator.yaml` | Source-controlled |
| `infra/compose/` | Service definitions, profiles, memory limits, published ports | `.env` | Source-controlled |
| `infra/postgres/schema.sql` | Tables `predictions`, `labels`, `data_batches`, `monitoring_results`, `retrain_events`, `deployments`, `events`, `gateway_state`, with the `RUNNING` partial unique index and `labels.exported_at`, `labels.exported_event_id` | `init` service | Source-controlled |
| `infra/prometheus`, `infra/grafana` | Scrape config, alert rules, provisioned datasources and dashboards | Prometheus, Grafana | Source-controlled |
| `scripts/` | Operational helpers called by the Makefile and CI | n/a | Source-controlled |
| `experiments/` | Stage 3 and Stage 6 research scripts (N-21, N-22); never imported by production code | `experiments/*` | Source-controlled |
| `tests/` | Test code and **synthetic** fixtures | `pyproject.toml` | Source-controlled |
| `docs/SPEC.md`, `docs/decision_log.md` | This specification and the register (Section C) saved as files | Project owner | Source-controlled |
| `docs/model_card.md` | Limitations, assumptions, criteria, failure cases; static text reviewed by hand at every release; per-version facts go to the MLflow artifact model\_card.md written by the `deploy` task at promotion (version, gate report digest, data version), never into git | Release process | Source-controlled |
| `docs/evidence/`, `docs/resource-report.md` | Recorded outputs of real runs (demo reports, tuning, resource measurements) | The agent or engineer who ran them | Source-controlled after review; never edited by hand |
| `.env`, MLflow and Postgres volumes, `.deployed_tag`, `images.tar` | Local runtime state | n/a | Generated, git-ignored |

`scripts/check_tree.py` fails if a top-level folder from this blueprint is missing or an unlisted top-level folder exists; it runs in `lint-test`.

## F. Phased execution plan

**Rules for every phase.** Start from a known testable state; create only the files the phase lists; run the listed commands; write the real command output to `docs/evidence/phase_<N>.md`; a phase is **done** only when every exit criterion is shown by recorded output. If a check fails: fix it, or stop and report; never weaken a test, threshold or gate. Effort is relative (S, M, L); no dates are given because no deadline or team size was supplied. **Feature freeze:** after Phase 10 exits, no extension (X-nn) may start until the Phase 12 demonstration is recorded.

**Order note.** The default order is kept except that the deployment and rollback module (D.8) is built in Phase 9, because retraining cannot be tested without it; Phase 11 then covers CI/CD and recovery tests in CI.

### Phase 0 – Environment baseline (S)

- **Objective / prerequisites.** Know the machine and pin versions. Needs Docker with Compose v2, git, Python 3.11.
- **Do.** Record `nproc`, `free -h`, `df -h`, `docker version`, `docker compose version` in `docs/resource-report.md`. Resolve and write `requirements/*.lock` (N-01), choosing the Airflow release whose official constraints file supports Python 3.11; record every chosen version in `docs/decision_log.md`.
- **Exit.** Report contains real measurements; lock files install in a clean virtualenv.
- **Recovery.** Under 8 GB RAM: plan on `make up-lite`; report the limitation to the owner (N-28).

### Phase 1 – Repository structure and configuration (S)

- **Prerequisites.** Phase 0.
- **Create.** `Makefile`, `pyproject.toml`, `.env.example`, `.gitignore`, `.gitleaks.toml`, `.pre-commit-config.yaml`, all `configs/*.yaml` and `profiles/`, `src/fairxai/{config,clock,db,notify,logs}.py`, `infra/compose/docker-compose.yml` (services `postgres`, `minio`, `init`), `infra/postgres/schema.sql`, `scripts/check_tree.py`, `.github/workflows/ci.yml` (job `lint-test` only), `docs/SPEC.md` and `docs/decision_log.md` (this document).
- **Requirements.** Every key in Section D has a typed field; unknown keys fail; profiles overlay; `clock.now()` honours the offset file; `notify()` per D.8.
- **Run.** `make env && make up-lite && make health`, `make lint`, `make test`.
- **Tests.** Config unknown-key rejection; profile overlay; clock offset; `notify()` with a local HTTP stub and with no webhook; schema applies twice without error; the `RUNNING` unique index rejects a second row.
- **Exit.** All pass; `check_tree.py` passes; postgres and minio healthy; buckets and databases exist; `lint-test` green in GitHub.
- **Recovery.** Port clash → change `.env`; schema error → fix `schema.sql`, `make reset`.

### Phase 2 – Dataset acquisition and validation (M)

- **Prerequisites.** Phase 1; the team's data file at `data/raw/`.
- **Create.** `scripts/prepare_raw.py`, `scripts/profile_data.py`, `src/fairxai/data/{load,prepare,split,validate,ingest}.py`, `pipelines/expectations/*.json`, `dvc.yaml`, `tests/data/*`, `tests/fixtures/make_synthetic.py`.
- **Run.** `python scripts/prepare_raw.py`, `dvc add`, `make data`, `python scripts/profile_data.py`.
- **Tests.** Suites pass on valid and fail on each deliberately invalid fixture (missing column, bad category, null, out-of-range, wrong row count); split disjointness, stratification on target x SEX, 9 stream batches of 500; ingest idempotency; rejected batch never reaches `accepted`.
- **Expected.** 30,000 rows, 22.1 % default rate in public data (verify). **Stop** and report if the profile contradicts the contract in D.2; update `data.yaml` only with an owner-approved decision.
- **Exit.** Tests pass; profile output saved to `docs/evidence/phase_2.md` including age-band and SEX x band group sizes; `dvc push` succeeded.
- **Recovery.** Hash mismatch → `dvc checkout`; missing remote bucket → rerun `init`.

### Phase 3 – Reproducible baseline training (M)

- **Prerequisites.** Phase 2.
- **Create.** `src/fairxai/features/pipeline.py` (first step: the D.2 category remap), `src/fairxai/models/{zoo,imbalance,tune,threshold,evaluate,train}.py` (steps 1–5 of D.3, results written to a local JSON), `tests/unit/models/*`, `tests/model/test_reproducibility.py`.
- **Run.** `make train` (profile `base`), then `FAIRXAI_PROFILE=test make train` twice.
- **Tests.** D.3 test list; reproducibility within 1e-6.
- **Exit.** Seven baselines ranked, top 3 tuned, threshold stored; wall-clock time and peak memory recorded in the resource report.
- **Recovery.** SVM too slow → the 900 s limit applies; if it dominates, report to the owner rather than removing it.

### Phase 4 – Experiment tracking and registry (M)

- **Prerequisites.** Phase 3.
- **Create.** `mlflow` service in compose; MLflow logging in `train.py`; `src/fairxai/registry/{registry,lineage}.py`; `tests/unit/registry/*`.
- **Verify the pinned MLflow** alias functions and `mlflow.sklearn` signature behaviour before coding against them (\[VERIFY\]).
- **Tests.** Alias rules (`set_candidate` refuses non-APPROVED; `promote_to_champion` and `deploy` refuse a REJECTED version; `promote_to_champion` ordering and consistency check); lineage resolve on a fixture run raises on any null.
- **Exit.** A run with all D.3 step-8 artifacts exists in MLflow and in MinIO; a registered version resolves via `lineage.resolve`.
- **Recovery.** Artifact upload fails → check `AWS_*` and endpoint; re-run is safe (new run).

### Phase 5 – Evaluation and fairness gate (M)

- **Prerequisites.** Phase 4.
- **Create.** `src/fairxai/fairness/{groups,metrics,calibration,report}.py`, `src/fairxai/registry/promotion.py`, `src/fairxai/explain/shap_global.py`; extend `train.py` with D.3 steps 5–8 (selection loop, test report, SHAP summary, registration with tags).
- **Tests.** The full D.4 test list, including exact-threshold boundaries and the aggregation truth table; selection loop picks the first APPROVED model in rank order.
- **Run.** `make train` on real data.
- **Expected.** A decision of `APPROVED`, `REJECTED` or `INDETERMINATE` with a complete gate report. **All three are valid outcomes.** If not `APPROVED`: save the report, do **not** alter any threshold, band or tolerance, notify the owner (R-1) and continue the remaining phases using the synthetic fixture champion (`fixture=true`, `gate_decision=FIXTURE_FORCED`, profiles `demo` and `test` only; built as `fixture_champion`, D.9) only for software testing.
- **Exit.** Tests pass; fairness report, gate report, SHAP summary and test-set report are in MLflow; version tags set; `candidate` alias set only when APPROVED.
- **Recovery.** Metric disagreement with a hand calculation → fix the code, never the fixture.

### Phase 6 – Single-model inference service (M)

- **Prerequisites.** Phase 5 (or the synthetic fixture champion).
- **Create.** `serving/` (app, Dockerfile), `gateway/`, `src/fairxai/deploy/probes.py`, compose services `serving-blue`, `serving-green`, `gateway`, `infra/prometheus/*`; `tests/api/*`.
- **Requirements.** D.6 in full except `/v1/explain`, which returns 501 until Phase 7 (the only permitted stub; the Phase 7 exit removes it); `X-Probe` handling per D.8 step 3.
- **Run.** `make up`, `POST /admin/load` on blue, predict through the gateway.
- **Tests.** The D.6 list except explain.
- **Exit.** API tests pass; p95 latency at 5 requests/s measured and recorded (N-12 is a target, the measurement is the evidence); Prometheus shows all metric names; a request with a bad key and a malformed body are rejected and not logged as predictions.
- **Recovery.** Model load failure → `GET /readyz` reports the reason; slot keeps serving previous model.

### Phase 7 – Explanation endpoint (S)

- **Create.** `src/fairxai/explain/shap_local.py`; `/v1/explain` in `serving/app/main.py`.
- **Verify** the SHAP explainer behaviour for the champion's model family against the pinned `shap` (\[VERIFY\]); explanation uses the loaded model only.
- **Tests.** Additivity within 1e-4; `method` and `output_space` truthful for tree, linear and kernel families (use tiny fixture models); `top_k` bounds; latency recorded.
- **Exit.** Explain tests pass on the deployed champion; no stub remains.

### Phase 8 – Monitoring and drift detection (M)

- **Prerequisites.** Phase 6; labels via `/v1/feedback`.
- **Create.** `src/fairxai/monitoring/{windows,drift,performance,triggers,run}.py`, Grafana provisioning and dashboards, `grafana` and `prometheus` completion in compose; `tests/unit/monitoring/*`.
- **Requirements.** D.7. Adapter first: write `compute_drift()` against the pinned Evidently; if the 1-day timebox fails, use the scipy contingency and record it in the decision log.
- **Tests.** The D.7 fixtures; persistence and reset rules.
- **Run.** Replay batches 1–2 with the simulator's replay only (no faults) and run `make monitor`.
- **Exit.** `monitoring_results` rows for every signal with correct `NA` handling; Grafana dashboards load from provisioning with data; drift-only fixtures produce no retrain request.
- **Recovery.** Adapter failure → `monitoring` task fails loudly, previous results retained.

### Phase 9 – Deployment, rollback and drift-triggered retraining (L)

- **Prerequisites.** Phases 6 and 8.
- **Create.** `src/fairxai/deploy/deploy.py`, `src/fairxai/retrain/retrain.py`, all seven DAGs in `pipelines/dags/`, `infra/airflow/Dockerfile`, Airflow services in compose, `src/fairxai/simulator/{replay,faults,fault_models,__main__}.py`; `tests/unit/{deploy,retrain}/*`.
- **Verify** the Airflow decorator import path for the pinned version (\[VERIFY\]); the DAG import test catches it.
- **Build in this order, testing each:** (a) `deploy()` and rollback against fixtures; (b) `train_and_register` and `deploy_candidate` DAGs; (c) throttle, acquire and export logic of retraining; (d) `retraining` and `monitoring` DAG wiring; (e) simulator faults.
- **Tests.** The D.8 test list: concurrency, each throttle reason, label consumption, stale `RUNNING`, insufficient labels, failed validation, rejected candidate keeps the champion, deploying a REJECTED version fails, `LOAD_FAILED`, `G6_FAILED`, G7 rollback, first deployment, approval on and off, manual rollback.
- **Exit.** All pass; a full cycle (drift fixture → request → retrain → decision → deploy or keep) runs through Airflow once, recorded.
- **Recovery.** A stuck `RUNNING` event → the stale cleanup marks it `FAILED`; a half-promoted registry → `RegistryInconsistent` stops the pipeline and `docs/runbook.md` lists the manual alias commands.

### Phase 10 – Automated integration testing (M)

- **Create.** `infra/compose/docker-compose.ci.yml`, `tests/integration/*`, markers `full` for Airflow-dependent tests.
- **Scenarios.** ingest → train → register → deploy → predict → explain → feedback → monitoring → retrain decision → rollback; each failure scenario of D.9 step 5; lineage; continuous requests during a flip with the non-2xx count recorded (must be 0); gateway restart keeps the active slot; a restarted slot reloads `EXPECTED_MODEL_VERSION` and serves again (and stays 503 `model_not_ready` when it is unset); purge removes only rows older than 90 days.
- **Run.** `make test-integration` locally from a clean state (`make reset`).
- **Exit.** All pass twice in a row from a clean start. **Feature freeze begins.**

### Phase 11 – CI/CD and deployment recovery (M)

- **Create.** Remaining jobs of `ci.yml`, `release.yml`, `scripts/{smoke.sh,deploy_code.sh}`, `.trivyignore` (reviewed entries only).
- **Verify** (\[VERIFY\]): gitleaks and Trivy action inputs, required-reviewer availability for the repository, hosted-runner memory for the integration job.
- **Tests.** A deliberately failing smoke test makes `deploy_code.sh` restore the previous tag and exit non-zero (run locally with a fixture image that fails health); the `release` job refuses when an upstream job failed.
- **Exit.** CI green on the final commit with all jobs; a release run with approval produced `images.tar`; resource report updated with real numbers.
- **Recovery.** If hosted runners cannot run the integration stack within memory, split it into smaller test selections; never skip the job.

### Phase 12 – Dashboards, alerts, demonstration polish (M)

- **Create.** Final Grafana dashboards, Prometheus rules check, `scripts/demo.sh`, `docs/model_card.md`, `docs/runbook.md`, `docs/architecture.md`, `README.md`, `docs/evidence/fault_tuning.md`.
- **Do.** Run the fault tuning step (D.9), then `make reset && make demo` and keep the generated evidence file.
- **Exit.** The final acceptance checklist in Section H is fully ticked with links to recorded evidence; any unticked item is listed as a known limitation in `docs/model_card.md` and in the final report.

## G. Coding-agent master execution prompt

Give this prompt to Cursor (or another coding agent) together with this document saved as `docs/SPEC.md`.

```markdown
# Role
You are implementing the FairXAI MLOps platform. `docs/SPEC.md` is the single source of truth: Section C (decision register) and Section D (implementation specification) are binding; Section E gives the repository layout; Section F gives the phases; Section H gives the acceptance checklist. You are an implementer, not an architect.

# Authority and limits
1. Do not change an approved decision (D1-D23, N-01..N-28, CR-1..CR-7) or the scope. If something seems wrong, stop and report it as a blocker with evidence; do not work around it.
2. Never invent API contracts, dataset columns, configuration keys, library features or file paths. Everything you need is in the spec. Where the spec says [VERIFY], check the behaviour against the installed pinned version and its official documentation before coding against it, and record what you found in `docs/decision_log.md`.
3. Do not add files, services, tools or dependencies that the spec does not list. Do not upgrade dependencies; use the lock files (N-01).
4. Do not implement extensions (X-nn) or anything in the extension backlog.
5. Keep every threshold, window, tolerance and policy in `configs/`. No duplicated sources of truth. Logic goes in `src/fairxai/`; DAGs only orchestrate.
6. Never hardcode or print secrets. Never disable, skip, loosen or bypass a failing test, validation, fairness gate or security check to get a green result. Never edit a test's expected values to match wrong code.
7. Never fabricate test results, logs, benchmark numbers, resource figures, dataset statistics or deployment success. Real numbers come only from commands you ran, and are written to `docs/evidence/` or `docs/resource-report.md`.
8. Test fixtures and anything synthetic must be named as synthetic. Fixture results are never reported as real measurements. Forced fixtures (`fixture=true`) are allowed only under profiles `demo` and `test`.
9. If a model fails the fairness gate (REJECTED or INDETERMINATE), that is a valid result: record it and report it. Do not tune thresholds, age bands, tolerances or the favorable label to make a model pass.

# Protocol (repeat for each phase, in order)
1. Read the whole of `docs/SPEC.md` once at the start, and the phase section again before each phase.
2. Inspect the repository and the environment (`git status`, tree, installed versions). Reuse existing correct code and conventions. Preserve unrelated working functionality.
3. State the current phase and confirm its prerequisites are satisfied by running the prerequisite checks. If not, stop.
4. Implement only the files and behaviour listed for the phase. Work in small steps; after each meaningful change run the targeted tests.
5. Run the phase's commands and tests. Fix regressions before moving on.
6. Write the real outputs (commands, results, changed files, failures, untested behaviour) to `docs/evidence/phase_<N>.md`.
7. Mark the phase done only when every exit criterion is demonstrated by recorded output. Then start the next phase. Never start a phase whose prerequisites are unmet.
8. Commit after each phase with a message naming the phase and listing tests that passed.

# Stop and report (do not guess) when
- a required decision is missing, contradictory or marked BLOCKING and the phase depends on it;
- the real dataset contradicts the contract in D.2;
- a pinned library behaves differently from what the spec assumes and the spec gives no contingency;
- a check fails and the fix would require changing a decision, threshold, gate or test expectation;
- no model is APPROVED (continue only software work on the fixture champion and tell the owner).

# Report format after every phase
Phase, files created or changed, commands run with exit codes, tests passed and failed (counts and names), measured values recorded, known failures, behaviour not tested, blockers, next phase and whether its prerequisites hold.

# Definition of done for the project
Every item of the final acceptance checklist (Section H) is ticked with a link to recorded evidence; unticked items are listed honestly as limitations.
```

## H. Verification, traceability and quality control

### H.1 Requirement-to-test traceability

Paths are under `tests/`. Evidence column = file the implementer must fill with real output; nothing is pre-filled.

| Req. | Requirement (MVP capability) | Verification | Measurable acceptance criterion | Evidence |
| --- | --- | --- | --- | --- |
| M-01 | Load and version data | `data/test_prepare.py`, `dvc status` | 30,000 rows; hashes equal `dvc.lock`; `dvc status` clean | `phase_2.md` |
| M-02 | Validate input data | `data/test_validate.py`, `data/test_ingest.py` | Every invalid fixture fails, valid passes; rejected batches never in `accepted/`; re-ingest is a no-op | `phase_2.md` |
| M-03 | Deterministic baseline training | `model/test_reproducibility.py`, `unit/models/*` | Two runs: same model, parameters, metrics within 1e-6; eval and test never used for selection | `phase_3.md` |
| M-04 | Track experiments | `integration/test_tracking.py` | Run holds every artifact listed in D.3 step 8 | `phase_4.md` |
| M-05 | Register model | `unit/registry/*` | Alias rules and consistency check hold; `lineage.resolve` has no null | `phase_4.md` |
| M-06 | Evaluate quality and fairness | `unit/fairness/*` | Hand-computed fixtures match; exact-threshold boundaries; ineligible-group and INDETERMINATE cases | `phase_5.md` |
| M-07 | Automated promotion gate | `unit/registry/test_promotion.py` | Aggregation truth table; candidate with higher PR-AUC but failing G3 is `REJECTED`; each gate returns each status; deploying a `REJECTED` version fails (`unit/deploy/`) | `phase_5.md` |
| M-08 | Single inference service | `api/*` | All D.6 contract tests pass; p95 latency at 5 requests/s measured and recorded | `phase_6.md` |
| M-09 | Prediction and explanation endpoints | `api/test_explain.py` | Additivity within 1e-4; `method` matches model family; no fallback explainer | `phase_7.md` |
| M-10 | Service and model monitoring data | `integration/test_monitoring.py` | All metric names scraped; `monitoring_results` has a row per signal per window | `phase_8.md` |
| M-11 | Drift detection | `unit/monitoring/*` | Results correct just below, at, above each threshold; drift-only never requests retraining | `phase_8.md` |
| M-12 | Controlled retraining | `unit/retrain/*`, `integration/test_retrain.py` | Two simultaneous requests give one `RUNNING` and one `SUPPRESSED`; each throttle reason fires; stale cleanup works | `phase_9.md` |
| M-13 | Re-evaluation of every candidate | `integration/test_retrain.py` | Every retrain event has a stored gate report from the same `decide()` | `phase_9.md` |
| M-14 | Rejection, recovery, rollback | `integration/test_recovery.py` | After each failure fixture, the gateway returns the previous version and 20 predictions succeed; manual rollback works; a restarted slot reloads `EXPECTED_MODEL_VERSION` | `phase_10.md` |
| Q-01 | Clean-start reproduction | Script run | `make reset && make up && make data && make train && make deploy` succeeds from a clean clone | `phase_10.md` |
| Q-02 | Lineage | `integration/test_lineage.py` | Deployed version resolves to run, commit, DVC hash, `data_version_id`, config, image tag, deployment | `phase_10.md` |
| Q-03 | No secrets exposed | gitleaks job; `api/test_auth.py` | gitleaks clean on full history; no key in logs or bodies | `phase_11.md` |
| Q-04 | CI/CD | GitHub run | All jobs of D.10 green on the final commit; `release` ran with approval | `phase_11.md` |
| Q-05 | Resource claims | `docs/resource-report.md` | Contains measured values from Phase 0 and Phase 11; estimates labelled | `phase_11.md` |
| Q-06 | Deterministic demo | `make demo` | Every D.9 step prints PASS; evidence file written and states which outcome occurred: *real-model demo passed* or *fixture-champion demo passed*; either passes Q-06 | `docs/evidence/demo_*.md` |
| Q-07 | Model card | Review | States limitations from D.4, data limits, R-1 to R-4, outcome of the real gate | `phase_12.md` |
| Q-08 | Centralized configuration | `unit/test_no_duplicate_thresholds.py` | The values 0.10, 0.15, 0.80, 0.45, 0.01, 0.05, 500, 300 do not appear as literals in `src/fairxai/fairness`, `registry/promotion.py`, `monitoring/triggers.py`, `retrain/` | `phase_5.md` |

### H.2 Final end-to-end acceptance checklist

- [ ] `docs/SPEC.md` and `docs/decision_log.md` match the final behaviour; every deviation is a recorded, owner-approved decision
- [ ] Phases 0–12 each have `docs/evidence/phase_<N>.md` with real command output
- [ ] All unit, data, model, API and integration tests pass twice in a row from a clean start
- [ ] Real-data training ran; the gate decision (APPROVED, REJECTED or INDETERMINATE) and its report are stored; if not APPROVED the owner has been told
- [ ] Champion serves through the gateway; `/v1/predict` and `/v1/explain` return valid responses
- [ ] Drift fixture, persistence rule, throttle and idempotency behave as in D.7 and D.8
- [ ] Each failure fixture leaves the previous champion serving; rollback verified
- [ ] Lineage test passes for the deployed model
- [ ] CI green on the final commit including security and Trivy; release job approved
- [ ] `docs/resource-report.md` holds measured numbers; model card lists limitations
- [ ] `make demo` evidence file shows all steps PASS and states whether it is a real-model demo passed or a fixture-champion demo passed
- [ ] Known failures and untested behaviour are listed honestly in the final report

### H.3 Quality-control pass 1: technical consistency

Checked: referenced sections and files exist; every service has a responsibility; interfaces agree across D.6, D.8 and the diagrams; configuration values are defined once; dependencies are ordered; terminology matches the glossary; acceptance criteria are measurable; commands match the Makefile and repository tree.

**Defects found and fixed in this document:** (1) age-band label `21-29` in N-04 contradicted D.4 (`18-29`); (2) the `X-Probe` header used by deployment probes was not in the API contract; (3) stale `RUNNING` retraining rows were cleaned only at startup, which could block acquisition until then; (4) `make test-integration` was referenced but undefined; (5) the model card was to be written into git by a task; it is now an MLflow artifact; (6) minimum labelled rows differed between N-10 and D.8; (7) G2 did not say how the champion is scored; (8) the champion's performance reference had no storage location (`eval_pr_auc` tag); (9) `auto_deploy` had no owning file; (10) the `init` container was missing from the inventory; (11) the hash used for the holdout split was unspecified; (12) the baseline thresholds in D.3 step 3 had no source. **Not defects, left as designed:** `/v1/explain` returns 501 only during Phase 6 (Phase 7 exit removes it).

### H.4 Quality-control pass 2: implementation-readiness simulation

Read as a coding agent opening an empty repository. For each phase the questions were: what to build, which files, which versions, which contracts, how to run, what success is, what to do on failure, what is left to guesswork.

- **Resolved by this pass:** deployment code was needed before retraining (Order note in F); the path taken when no model passes the gate (Phase 5); which data feeds drift and performance windows (D.9: one batch per window); where synthetic fixtures are allowed (D.3, D.9).
- **Residual items an agent must still handle, by procedure and not by guessing:** choosing exact pinned versions in Phase 0 under N-01 (recorded in the decision log); every \[VERIFY\] behaviour (Airflow import path, MLflow alias functions, Great Expectations 1.x API, Evidently result structure, SHAP explainer types, Trivy and gitleaks action inputs, required reviewers); any contradiction between the real dataset and D.2 (stop rule).

**Readiness verdict.** Phases 0 and 1 can start now. Later phases can proceed on the stated defaults, but this specification is **not declared fully implementation-ready** until the owner (a) supplies the Stage 1 code and file (N-02), (b) confirms or changes the fairness defaults N-03, N-04, N-05, N-08, and (c) ratifies or rejects CR-1 to CR-7. Hardware compatibility is unverified until Phase 0 and Phase 11 measurements exist (N-28).

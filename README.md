# FairXAI MLOps

An end-to-end MLOps platform for a **fair and explainable credit-default model**, built on the public UCI
*Default of Credit Card Clients* (Taiwan, 2005) dataset. It covers versioned data, validated ingestion,
reproducible training and tuning, a **hard fairness gate** in front of every promotion, an MLflow model
registry, blue-green serving with SHAP explanations, and drift and performance monitoring with Grafana
dashboards.

The binding specification is [`docs/SPEC.md`](docs/SPEC.md). The project is built phase by phase
(SPEC Section F), and every phase has a write-up of real command output in [`docs/evidence/`](docs/evidence/).

---

## Contents
- [Status](#status)
- [Results so far](#results-so-far)
- [Architecture](#architecture)
- [Repository layout](#repository-layout)
- [Prerequisites](#prerequisites)
- [Setup from a fresh clone](#setup-from-a-fresh-clone)
- [Using the system](#using-the-system)
- [Tests and CI](#tests-and-ci)
- [Configuration](#configuration)
- [Continuing the project (Phases 9–12)](#continuing-the-project-phases-912)
- [Troubleshooting](#troubleshooting)
- [Known limitations](#known-limitations)

---

## Status

| Phase | Scope | State | Evidence |
| --- | --- | --- | --- |
| 0 | Environment baseline, pinned versions | Done | [phase_0](docs/evidence/phase_0.md) |
| 1 | Repo structure, typed config, Postgres schema, CI lint/test | Done | [phase_1](docs/evidence/phase_1.md) |
| 2 | Data contract, Great Expectations validation, ingestion, splits, DVC | Done | [phase_2](docs/evidence/phase_2.md) |
| 3 | 7 baselines, Optuna tuning, thresholds, reproducibility | Done | [phase_3](docs/evidence/phase_3.md) |
| 4 | MLflow tracking, registry aliases, lineage | Done | [phase_4](docs/evidence/phase_4.md) |
| 5 | Fairness metrics, promotion gate G1–G5, SHAP summary | Done | [phase_5](docs/evidence/phase_5.md) |
| 6 | FastAPI serving slots, blue-green gateway, Prometheus | Done | [phase_6](docs/evidence/phase_6.md) |
| 7 | `/v1/explain` with SHAP | Done | [phase_7](docs/evidence/phase_7.md) |
| 8 | Monitoring, Evidently drift, triggers, Grafana | Done | [phase_8](docs/evidence/phase_8.md) |
| 9 | Deployment + rollback, retraining, Airflow DAGs, simulator faults | **Next** | – |
| 10 | Integration tests | Not started | – |
| 11 | Full CI/CD (security, Trivy, release), deploy-code script | Not started | – |
| 12 | Dashboard polish, model card, `make demo` | Not started | – |

Design decisions taken during implementation, and every `[VERIFY]` finding against pinned libraries,
are in [`docs/decision_log.md`](docs/decision_log.md). Measured resource figures are in
[`docs/resource-report.md`](docs/resource-report.md).

## Results so far

All numbers are from real runs on the UCI data; details are in the evidence files.

| Item | Value |
| --- | --- |
| Data | 30,000 rows, 22.12 % default rate; splits train 18,000 / eval 4,500 / test 3,000 / stream 9 × 500 |
| Selected model | XGBoost (best of 7 baselines → top 3 tuned with 25 Optuna trials each) |
| Gate decision | **APPROVED** (G1–G5 all PASS); registered as `fairxai-credit-default` v1 |
| Eval PR-AUC / test PR-AUC | 0.550 / 0.570 (floor 0.45), decision threshold 0.53 |
| Fairness on eval (favorable = predicted *not* to default) | EO diff sex 0.027, age 0.048, sex×age 0.075; DI sex 0.96, age 0.92, sex×age 0.88 |
| Serving latency (5 req/s via gateway) | predict p95 35.8 ms, explain p95 45.7 ms, 0 errors |
| Tests | 258 passed (`pytest tests`, with Postgres running) |

Passing these thresholds does **not** prove the model is fair; they are conventional engineering
tolerances, not legal standards (SPEC D.4).

## Architecture

```text
 data/raw (DVC, MinIO remote) -> prepare -> split -> train | eval | test | stream batches
                                                     |
             python -m fairxai.models.train --track  v
   +--------+       +---------+      +-----------+      gate G1-G5 -> alias "candidate"
   | minio  |<----->| mlflow  |<---->| registry  |
   +--------+       +---------+      +-----------+
                         ^ /admin/load
              +----------+-----------+
              |                      |
        serving-blue           serving-green        (FastAPI, one model each)
              ^                      ^
              +------ gateway -------+   :8080  /v1/predict  /v1/explain  /v1/feedback
                         ^
   clients / simulator --+        predictions + labels -> postgres -> monitoring job
                                                                          |
                 prometheus (:9090) <- /metrics        grafana (:3000) <--+
```

| Service | Port (127.0.0.1) | Purpose |
| --- | --- | --- |
| postgres | 5432 | databases `fairxai` (prediction log, monitoring, events), `mlflow`, `airflow` |
| minio | 9000 / 9001 | S3 buckets `mlflow` (artifacts) and `dvc` (data remote) |
| mlflow | 5000 | tracking server and model registry |
| serving-blue / serving-green | 8001 / 8002 | model slots |
| gateway | 8080 | routes `/v1/*` to the active slot, blue-green flip |
| prometheus | 9090 | metrics and alert rules |
| grafana | 3000 | dashboards: Service, Model, Fairness, Lifecycle |

## Repository layout

```text
configs/          all thresholds, windows, policies (validated; unknown keys fail)
  profiles/       demo.yaml, test.yaml overlays (FAIRXAI_PROFILE)
data/             raw/*.dvc pointer; processed/ and stream/ are DVC outputs (git-ignored)
src/fairxai/      all logic: data/, features/, models/, fairness/, explain/, registry/, monitoring/,
                  deploy/, simulator/, config.py, db.py, notify.py, httpapi.py
serving/          slot FastAPI app + Dockerfile
gateway/          gateway FastAPI app + Dockerfile
infra/            compose/, postgres/schema.sql, mlflow/, prometheus/, grafana/
pipelines/        expectations/ (Great Expectations suites); dags/ arrive in Phase 9
scripts/          prepare_raw.py, profile_data.py, make_env.py, init_stack.py, check_tree.py
tests/            unit/ data/ model/ api/ (+ integration/ in Phase 10); fixtures are SYNTHETIC
docs/             SPEC.md, decision_log.md, resource-report.md, evidence/
```

## Prerequisites

| Tool | Notes |
| --- | --- |
| Git | |
| Docker Desktop (Compose v2) | Give it **at least 8 GB** of memory; the current stack idles at about 2.5 GB, and Airflow (Phase 9) adds about 2 GB |
| Python **3.11** exactly | e.g. `uv python install 3.11` ([uv](https://docs.astral.sh/uv/)) |
| `make` (optional) | Not installed on Windows by default. Every target is a short command in the [`Makefile`](Makefile) you can run by hand |

## Setup from a fresh clone

Commands are for **Git Bash / Linux / macOS**. PowerShell equivalents are given where they differ.

### 1. Python environment
```bash
git clone https://github.com/inslot2525-ctrl/MLOPS_CP.git
cd MLOPS_CP
uv venv --python 3.11 .venv
uv pip install -r requirements/dev.lock
uv pip install --no-deps -e .
source .venv/bin/activate          # Windows Git Bash: source .venv/Scripts/activate
                                   # PowerShell:      .venv\Scripts\Activate.ps1
```

### 2. Secrets file
```bash
python scripts/make_env.py         # = make env
```
This copies `.env.example` to `.env` (git-ignored) and generates a fresh admin key and simulator key
plus their SHA-256 hashes. The keys are never printed; read them from `.env` when you need them.
Replace the other example passwords if the machine is shared.

Load `.env` into your shell before running host-side commands:
```bash
set -a; source .env; set +a
export PYTHONIOENCODING=utf-8 MLFLOW_DISABLE_AGENT_HINT=1
```
```powershell
Get-Content .env | Where-Object { $_ -match '^[A-Z_]+=' } | ForEach-Object {
  $k, $v = $_ -split '=', 2; Set-Item "env:$k" $v }
$env:PYTHONIOENCODING = "utf-8"; $env:MLFLOW_DISABLE_AGENT_HINT = "1"
```

### 3. Data
Each clone builds its own copy; `dvc pull` only works against your own MinIO after a first `dvc push`.
```bash
curl -L -o uci.zip "https://archive.ics.uci.edu/static/public/350/default+of+credit+card+clients.zip"
unzip uci.zip                                      # -> "default of credit card clients.xls"
python scripts/prepare_raw.py "default of credit card clients.xls"   # -> data/raw/credit_default.csv
dvc repro                                          # prepare + split (seeded, deterministic)
dvc status                                         # should say: Data and pipelines are up to date
python scripts/profile_data.py                     # sanity check: 30000 rows, default_rate=0.2212
```
The splits are deterministic, so the file hashes should match the committed `dvc.lock`.

### 4. Start the stack
```bash
make up        # or, without make:
C="docker compose --env-file .env -f infra/compose/docker-compose.yml --profile core"
$C up -d --wait postgres minio
$C run --rm init                     # creates databases, schema, buckets (idempotent)
$C up -d --build --wait mlflow serving-blue serving-green gateway prometheus grafana
$C ps
```
First build takes a few minutes (serving and gateway images are about 2.4 GB). Optional, after MinIO is
up: `dvc push` stores the data in your MinIO bucket `dvc`.

### 5. Train and register a model
```bash
python -m fairxai.models.train --track             # = make train; about 20-25 min (SVM tuning dominates)
```
This prints the gate decision and the registered version. An `APPROVED` version gets alias
`candidate`; `REJECTED` and `INDETERMINATE` are valid outcomes and are never promoted. Inspect the run at
http://127.0.0.1:5000. For a quick software check on SYNTHETIC data, run the tests instead (see below).

### 6. Load the model and call the API
```bash
# load version 1 into the blue slot (admin key from .env)
curl -s -X POST http://127.0.0.1:8001/admin/load -H "X-API-Key: $ADMIN_API_KEY" \
     -H "Content-Type: application/json" -d '{"model_version": 1}'
curl -s http://127.0.0.1:8080/readyz
```
```bash
BODY='{"features": {"LIMIT_BAL": 20000, "SEX": 2, "EDUCATION": 2, "MARRIAGE": 1, "AGE": 24,
  "PAY_0": 2, "PAY_2": 2, "PAY_3": -1, "PAY_4": -1, "PAY_5": -2, "PAY_6": -2,
  "BILL_AMT1": 3913, "BILL_AMT2": 3102, "BILL_AMT3": 689, "BILL_AMT4": 0, "BILL_AMT5": 0, "BILL_AMT6": 0,
  "PAY_AMT1": 0, "PAY_AMT2": 689, "PAY_AMT3": 0, "PAY_AMT4": 0, "PAY_AMT5": 0, "PAY_AMT6": 0}}'
curl -s -X POST http://127.0.0.1:8080/v1/predict -H "X-API-Key: $SIM_API_KEY" -H "Content-Type: application/json" -d "$BODY"
curl -s -X POST "http://127.0.0.1:8080/v1/explain?top_k=5" -H "X-API-Key: $SIM_API_KEY" -H "Content-Type: application/json" -d "$BODY"
```

## Using the system

### API (full contract: SPEC D.6)
| Endpoint | Auth | Purpose |
| --- | --- | --- |
| `POST /v1/predict` | client key | prediction, `probability_default`, `decision_threshold`; logged unless `X-Probe: true` |
| `POST /v1/explain?top_k=1..23` | client key | SHAP contributions per input feature; `method` and `output_space` stated |
| `POST /v1/feedback` | client key | `{"labels": [{"request_id": ..., "default": 0/1}]}`, 1–1000 per call |
| `GET /v1/model` | client key | loaded version, run id, threshold, git commit, data version |
| `GET /healthz`, `/readyz`, `/metrics` | none | liveness, readiness, Prometheus |
| `POST /admin/load` (slot) | admin key | load a registered version; on failure the slot keeps its previous model |
| `GET /admin/state`, `POST /admin/active` (gateway) | admin key | inspect / flip the active slot (409 if the target slot is not ready) |

All errors use one body: `{"error": {"code", "message", "request_id", "details"}}`.

### Monitoring
Run one monitoring window (the last 500 logged predictions of the newest model version):
```bash
python -m fairxai.monitoring.run              # add --engine scipy to bypass Evidently
```
It writes one row per signal to `monitoring_results` (`data_quality`, `feature_drift`,
`prediction_drift`, `performance`, `fairness`) with status `OK`/`WARN`/`ALERT`/`NA`, and evaluates the
retraining triggers. Batch replay helpers are in `src/fairxai/simulator/replay.py` (the simulator CLI
and fault scenarios arrive in Phase 9).

### Dashboards
- Grafana: http://127.0.0.1:3000 (user `admin`, password `GRAFANA_ADMIN_PASSWORD` from `.env`), folder **FairXAI**
- Prometheus: http://127.0.0.1:9090 (targets, the 5 alert rules)
- MLflow: http://127.0.0.1:5000 · MinIO console: http://127.0.0.1:9001

## Tests and CI

```bash
ruff check . && ruff format --check . && python scripts/check_tree.py      # = make lint
pytest tests                                                               # = make test (+ model tests)
```
- Tests never use the real dataset; fixtures are **synthetic** (`tests/fixtures/make_synthetic.py`).
- Database tests run when `FAIRXAI_TEST_DB_URL` is set, e.g.
  `export FAIRXAI_TEST_DB_URL=postgresql://fairxai:<POSTGRES_PASSWORD>@127.0.0.1:5432/fairxai`;
  otherwise they are skipped (and reported as skipped, not passed).
- Registry and tracking tests use a real local MLflow (sqlite in a temp dir), not mocks.
- GitHub Actions (`.github/workflows/ci.yml`) currently runs the `lint-test` job on every push. Security
  scans, image builds, Trivy, the integration stack and the release job are Phase 11.

## Configuration

Every threshold lives in `configs/` and is validated by `src/fairxai/config.py`; no threshold is
hard-coded (a test enforces it). Key defaults (SPEC Section C; owner confirmation still requested for
N-03, N-04, N-05, N-08):

| Setting | Value | File |
| --- | --- | --- |
| Favorable outcome | predicted *not* to default (`favorable_label: 0`) | fairness.yaml |
| Age bands | 18-29, 30-39, 40+ | fairness.yaml |
| Hard gates | EO diff ≤ 0.10 (sex, age), ≤ 0.15 (sex×age); DI ≥ 0.80 | fairness.yaml |
| Group eligibility | ≥ 100 rows and ≥ 50 in the metric denominator | fairness.yaml |
| PR-AUC | floor 0.45; challenger must beat champion by ≥ 0.01 | gates.yaml |
| Monitoring window | last 500 predictions; ≥ 300 rows / labelled rows required | base.yaml, monitoring.yaml |
| Retraining | drift in 3 consecutive windows or performance drop in 2; cooldown 60 min; ≤ 3 per week | retraining.yaml |

Select a profile with `FAIRXAI_PROFILE=demo|test` (shorter cooldowns, 3 Optuna trials for `test`).

## Continuing the project (Phases 9–12)

Anyone can continue this by hand or with any coding assistant. Claude Code is not required.

1. Read [`docs/SPEC.md`](docs/SPEC.md): Section C (decisions) and D (contracts) are binding, Section E is
   the file layout, **Section F lists each phase's files, tests and exit criteria**, Section H is the final
   checklist. Section G is a ready-made prompt if you use an AI coding assistant.
2. Read the latest evidence file and [`docs/decision_log.md`](docs/decision_log.md) to see what already
   deviates from the spec and why.
3. Rules that keep the project honest:
   - write real command output to `docs/evidence/phase_<N>.md`; never invent numbers;
   - never loosen a threshold, gate or test to get a green result;
   - fixtures and synthetic data must be labelled as such;
   - record every deviation from the spec in `docs/decision_log.md`;
   - commit once per phase, naming the phase and the tests that passed.

What Phase 9 needs next (SPEC F, D.8, D.9): `src/fairxai/deploy/deploy.py` (G6 live-slot check, flip,
G7 observation, automatic rollback), `src/fairxai/retrain/retrain.py` (acquire / throttle / export /
train / decide), the seven DAGs in `pipelines/dags/`, `infra/airflow/Dockerfile`, Airflow services in
compose, and `src/fairxai/simulator/{__main__,faults,fault_models}.py`.

## Troubleshooting

| Symptom | Fix |
| --- | --- |
| `UnicodeEncodeError ... '\U0001f3c3'` during training (Windows) | `export PYTHONIOENCODING=utf-8` (the CLIs also force UTF-8) |
| `pull access denied for minio/minio` | Expected: the compose file uses `cgr.dev/chainguard/minio` instead |
| Slot `/readyz` 503 with "Invalid Host header" | MLflow host check; compose passes `--allowed-hosts`. Rebuild/restart `mlflow` |
| Slot `/readyz` 503 "no model loaded" | `POST /admin/load {"model_version": N}` with the admin key |
| Gateway flip returns 409 | Target slot has no model; load one first |
| MLflow cannot write artifacts | Check `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `MLFLOW_S3_ENDPOINT_URL` in your shell |
| Port already in use | Override `POSTGRES_PORT`, `MLFLOW_PORT`, `GATEWAY_PORT`, `GRAFANA_PORT`, etc. in `.env` |
| `dvc pull` fails on a new clone | Normal: rebuild the data (Setup step 3), then `dvc push` to your own MinIO |

## Known limitations

- Version 1 of the model was trained from uncommitted code (`git_dirty=true` in its lineage); a release
  needs a clean retrain (SPEC D.5).
- Live fairness on 500-row windows is noisy: window 2 flagged `eo_age` 0.155 against 0.048 on the
  4,500-row eval set (risk R-2). Alerts notify only and never retrain by themselves.
- The Taiwan data has no timestamps, so monitoring windows follow prediction-log order (A-10).
- The Stage 1 preprocessing equivalence (N-02) is unverified: the public UCI file was used.
- Serving images are large (about 2.4 GB) because they include SHAP and the full scientific stack.
- Measured on one Windows host with 7.5 GB for Docker; see `docs/resource-report.md`.

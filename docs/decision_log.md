# Decision log

The authoritative register is Section C of [SPEC.md](SPEC.md). This file records implementation-time decisions and `[VERIFY]` findings.

| Date | Phase | Decision / finding | Evidence |
| --- | --- | --- | --- |
| 2026-10-16 | 0 | Python 3.11.15 installed with `uv python install 3.11` (N-01). Host had only 3.12-3.14. | `docs/evidence/phase_0.md` |
| 2026-10-16 | 0 | Docker CLI present but daemon was not running; stack-level checks deferred. | `docs/evidence/phase_0.md` |
| 2026-10-16 | 1 | `init` is a python:3.11-slim one-shot running `scripts/init_stack.py` (psycopg + boto3), so one container handles both Postgres and MinIO setup. | compose file |
| 2026-10-16 | 1 | Compose services carry profile `core`; `make up` starts postgres+minio and runs `init`. | Makefile |
| 2026-10-16 | 2 | Taiwan data fetched from public UCI (dataset 350) because the team file was unavailable; N-02 stays open. | `docs/evidence/phase_2.md` |
| 2026-10-16 | 2 | `reference_stats.json` written by the `split` stage (needs the train partition). | `dvc.yaml` |
| 2026-10-16 | 2 | Outlier `f_ref` defined as flagged-cell fraction over the 13 continuous columns. | `src/fairxai/data/split.py` |
| 2026-10-16 | 2 | [VERIFY] Great Expectations 1.24.0: ephemeral context + `add_dataframe_asset` + `batch.validate(suite)` works; DVC 3.67.1. | `src/fairxai/data/validate.py` |
| 2026-10-16 | 3 | SVM `C`/`gamma` search ranges chosen (spec silent): C 0.1-100, gamma 1e-4..1e-1, log. sklearn 1.9 deprecates `SVC(probability=True)`; kept per spec. | `configs/training.yaml` |
| 2026-10-16 | 3 | Estimators fit single-threaded; parallelism is across CV folds (`training.n_jobs`), for determinism. | `src/fairxai/models/zoo.py` |
| 2026-10-16 | 4 | [VERIFY] MLflow 3.17.1: alias API OK; `log_model(name=...)` yields LoggedModel; skops default rejected custom types, so `serialization_format="cloudpickle"` is used. | `docs/evidence/phase_4.md` |
| 2026-10-16 | 4 | `Registry.register_version` takes a model URI rather than run_id (MLflow 3 LoggedModel). Versions are tagged `gate_decision=PENDING_GATE` until the Phase 5 gate exists. | `src/fairxai/registry/registry.py` |
| 2026-10-16 | 5 | `minio/minio` is no longer pullable from Docker Hub (and quay.io returns 401). Switched to `cgr.dev/chainguard/minio:latest` (MinIO RELEASE.2026-09-22); healthcheck uses bash `/dev/tcp` since the image has no curl. MinIO itself (D4) is unchanged. | compose file |
| 2026-10-16 | 5 | MLflow server image installs `psycopg[binary]` (SQLAlchemy 2.x defaults the postgresql dialect to psycopg 3); `--workers 1` cuts idle memory from 2.21 to 1.68 GiB. | `infra/mlflow/Dockerfile` |
| 2026-10-16 | 5 | Docker Desktop VM has 7.48 GiB, below the 16 GB provisional budget (N-28). | `docs/resource-report.md` |
| 2026-10-16 | 5 | Added config keys `gates.min_eval_rows: 100`, `fairness.calibration_bins: 10`, `fairness.decimals: 6` so no threshold is a literal in code (Q-08). | configs |
| 2026-10-16 | 5 | [VERIFY] shap 0.51.0 and fairlearn 0.15.0. XGBoost TreeExplainer output is log-odds; sklearn tree models use `model_output="probability"` with a 100-row background. | `src/fairxai/explain/shap_global.py` |
| 2026-10-16 | 5 | G1 failure on the training frame raises (training stops with an error) rather than registering a REJECTED version: there is no valid model to register from invalid data. | `src/fairxai/models/train.py` |
| 2026-10-16 | 5 | Train CLI reconfigures stdout/stderr to UTF-8 (MLflow emoji crashed `end_run` on cp1252). Failed run `ae5a495f` marked FAILED. | `docs/evidence/phase_5.md` |
| 2026-10-16 | 5 | `CategoryRemap.fit` records `feature_names_in_` so the pipeline exposes its input schema to G5. | `src/fairxai/features/pipeline.py` |
| 2026-10-16 | 6 | Added `src/fairxai/httpapi.py` (key hashing, error body) shared by slots and gateway, and `scripts/make_env.py` for `make env` key generation; neither file is in the Section E tree. | repo |
| 2026-10-16 | 6 | Gateway is given client key hashes only so a client key on an admin route returns 403 (not 401). | `gateway/app/main.py` |
| 2026-10-16 | 6 | `requirements/serving.lock` pins scikit-learn 1.9.1, xgboost 3.2.0, pandas 3.0.6, numpy 2.4.6 to the training versions; uses `mlflow-skinny==3.17.1`. Gateway reuses the serving lock (one image layer set). | `requirements/serving.in` |
| 2026-10-16 | 6 | [VERIFY] MLflow 3.17 server validates the Host header; `--allowed-hosts` lists the internal service name and localhost. | compose file |
| 2026-10-16 | 6 | Slot maps only MLflow `RESOURCE_DOES_NOT_EXIST` to 404; any other registry error is 500 and keeps the previous model. | `serving/app/model_store.py` |
| 2026-10-16 | 7 | `/v1/explain` contributions are per raw request feature (encoded SHAP values summed per source column); additivity holds. Explainer is built at `/admin/load`. Training logs `explain_background.csv` (1,000 seeded train rows). | `src/fairxai/explain/shap_local.py` |

# Phase 4 evidence – experiment tracking and registry

## What was built
`src/fairxai/registry/{registry,lineage}.py`, `src/fairxai/models/tracking.py` (run logging, registration, G5 registered-URI smoke check), `--track` flag in `train.py`, `mlflow` service in compose (`infra/mlflow/Dockerfile`, Postgres backend, MinIO artifact root), `tests/fixtures/smoke_set.csv` (synthetic).

## [VERIFY] findings on the pinned MLflow 3.17.1
- Alias API exists and works: `set_registered_model_alias`, `delete_registered_model_alias`, `get_model_version_by_alias`, `set_model_version_tag`.
- `mlflow.sklearn.log_model` now takes `name=` (not `artifact_path`); the model is a *LoggedModel*, not a run artifact under `model/`.
- Default serialization is `skops`, which refuses custom/tree types unless trusted. We log with `serialization_format="cloudpickle"` (our own artifacts, our own registry).
- `register_version` takes a `model_uri` (the logged model's URI) instead of `run_id`.

## Tests (real local MLflow with sqlite + tmp artifact dir; no mocks)
- `tests/unit/registry/test_registry.py` – 9 tests: `set_candidate` refuses REJECTED/INDETERMINATE/PENDING; `promote_to_champion` refuses REJECTED; previous/champion/candidate ordering; rollback to a previously held champion stays allowed; repeated promote stable; `FIXTURE_FORCED` only under `demo`/`test`; `RegistryInconsistent` on a dropped alias write; `lineage.resolve` returns fields and raises on null / missing deployment fields.
- `tests/model/test_tracking.py` – end-to-end on synthetic data: artifacts (`config_snapshot.json`, `optuna_summary.json`, `baselines.csv`, `results.json`), version tags, G5 smoke PASS from the registered URI, no `candidate` alias, 75 nested trial runs, `lineage.resolve` OK.

Final run: `ruff check`, `ruff format --check`, `check_tree.py` clean; **`pytest tests`: 66 passed, 2 skipped** (the 2 skipped are the Postgres schema tests; Docker daemon off).

## Exit criteria
- [x] A run with the artifacts that exist at this phase is logged and a registered version resolves via `lineage.resolve` (**local sqlite/file MLflow only**).
- [ ] The run in **MLflow on Docker + artifacts in MinIO** – NOT verified: Docker daemon off. Compose service is written but never started.
- [ ] Fairness report, gate report and SHAP summary artifacts – Phase 5 (gate_decision is `PENDING_GATE`, no `candidate` alias is ever set before then).
- Registry events are written to the `events` table through `db_event_sink`; with no database the sink logs a warning and continues (untested against live Postgres).

## Other changes
- `requirements/dev.lock` is now universal (`uv pip compile --universal`, Python 3.11) with the full training stack; CI installs from it.

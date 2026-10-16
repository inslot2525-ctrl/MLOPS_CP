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

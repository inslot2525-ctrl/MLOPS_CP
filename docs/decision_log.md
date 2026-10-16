# Decision log

The authoritative register is Section C of [SPEC.md](SPEC.md). This file records implementation-time decisions and `[VERIFY]` findings.

| Date | Phase | Decision / finding | Evidence |
| --- | --- | --- | --- |
| 2026-10-16 | 0 | Python 3.11.15 installed with `uv python install 3.11` (N-01). Host had only 3.12-3.14. | `docs/evidence/phase_0.md` |
| 2026-10-16 | 0 | Docker CLI present but daemon was not running; stack-level checks deferred. | `docs/evidence/phase_0.md` |
| 2026-10-16 | 1 | `init` is a python:3.11-slim one-shot running `scripts/init_stack.py` (psycopg + boto3), so one container handles both Postgres and MinIO setup. | compose file |
| 2026-10-16 | 1 | Compose services carry profile `core`; `make up` starts postgres+minio and runs `init`. | Makefile |

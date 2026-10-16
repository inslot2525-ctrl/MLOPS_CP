# Phase 1 evidence – repository structure and configuration

## Commands and results (this session, Python 3.11.15 venv)
| Command | Result |
| --- | --- |
| `ruff check .` | All checks passed |
| `ruff format .` | 16 files unchanged after formatting |
| `python scripts/check_tree.py` | exit 0 |
| `pytest tests` | **11 passed, 2 skipped** |

## Skipped tests (not faked)
`tests/unit/test_schema.py` (schema applies twice; `RUNNING` unique index rejects a second row) needs a real Postgres via `FAIRXAI_TEST_DB_URL`. Docker was not running, so these two were **not executed**.

## Tests that ran
Config load, profile overlay (`demo`, `test`), unknown-key rejection, unknown profile-section rejection, clock offset/advance/reset, `notify()` with no DB or webhook, slack/discord webhook bodies against a local HTTP stub, webhook failure not raising, log redaction, `check_tree`.

## Exit criteria status
- [x] Unit tests pass, `check_tree.py` passes.
- [ ] postgres and minio healthy; buckets/databases exist — **not verified** (Docker daemon off).
- [ ] `lint-test` green in GitHub — pending first push.
- Not yet verified: `make up-lite`, `make health`.

## Deviations
- `ci.yml` installs with `pip install -e ".[dev]"` rather than from lock files (N-01 asks for lock-file installs); to be tightened in Phase 11 when all locks exist.
- Committed `.env.example` API key hashes are zeros; `make env` does not yet generate real keys/hashes (needed in Phase 6).

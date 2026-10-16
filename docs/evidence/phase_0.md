# Phase 0 evidence – environment baseline

Measured 2026-10-16 on the owner's Windows 11 host (commands run in this session).

| Item | Result |
| --- | --- |
| Logical CPUs (`nproc`) | 14 |
| Physical RAM (`systeminfo`) | 15,814 MB |
| Free disk on D: (`df -h /d`) | 72 GB free of 112 GB |
| Docker client | 29.8.2 |
| Docker Compose | v5.6.0 |
| Docker daemon | **not running** (`docker info` could not connect to `dockerDesktopLinuxEngine`) |
| Python | 3.11.15 installed via `uv python install 3.11` (host previously had 3.12–3.14 only) |
| git | 2.54.0 |

## Locks
`requirements/dev.lock` compiled with `uv pip compile requirements/dev.in --python-version 3.11` (61 lines). The `app`, `airflow` and `serving` locks are not yet produced: they are created in the phases that introduce those dependencies (Airflow lock needs its official constraints file, chosen in Phase 9).

## Not done / limitations
- No container could be started because the Docker daemon was off. Container memory and start-time figures are **not measured**; `docs/resource-report.md` is deferred until the daemon is available.
- The stated exit criterion "lock files install in a clean virtualenv" is demonstrated only for `dev.lock`'s source (`pip install -e ".[dev]"` into a fresh uv venv succeeded).

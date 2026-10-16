# Resource report

All numbers below were measured on the owner's machine in this session. Anything not measured is marked so.

## Host (Phase 0, 2026-10-16)
| Item | Value | Source |
| --- | --- | --- |
| Logical CPUs | 14 | `nproc`; Docker reports 14 |
| Host RAM | 15,814 MB | `systeminfo` |
| **Memory available to Docker** | **7.48 GiB** | `docker info` (Docker Desktop VM limit) |
| Free disk on D: | 72 GB of 112 GB | `df -h` |
| Docker / Compose | 29.8.2 / v5.6.0 | `docker version`, `docker compose version` |

The Docker VM limit (7.48 GiB) is below the spec's provisional 16 GB budget and near its 8 GB minimum (N-28). Airflow (estimated 1.5–2 GB) has not been started yet; `make up-lite`-style operation may be needed later.

## Container memory (`docker stats --no-stream`, idle)
| Container | Memory | When |
| --- | --- | --- |
| postgres | 73–78 MiB | Phase 5, idle |
| minio (cgr.dev/chainguard/minio) | 82–88 MiB | Phase 5, idle |
| mlflow (default workers) | 2.21 GiB | Phase 5, right after start |
| mlflow (`--workers 1`) | 1.68 GiB | Phase 5, right after start |

## Start-up times (wall clock)
| Step | Seconds |
| --- | --- |
| `up --wait postgres minio` (images cached) | 11 |
| `run --rm init` (pip install inside python:3.11-slim) | about 40 (not timed precisely) |
| MLflow image build + start | 83–93 (first build) |

## Training (Phase 3, real data, host Python, not in Docker)
| Item | Value |
| --- | --- |
| Total wall clock | 1,130 s |
| SVM baseline (5-fold, probability=True) | 180 s |
| SVM tuning (25 trials) | 812 s |
| Random forest tuning | 91 s |
| XGBoost tuning | 17 s |
| Peak memory | not measured |

## Not yet measured
Serving p95 latency at 5 requests/s (Phase 6), Airflow memory (Phase 9), Phase 11 re-measurement.

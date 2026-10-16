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
| serving-blue (model v1 loaded, after 600 requests) | 207 MiB of 1 GiB limit | Phase 6 |
| serving-green (no model) | 75 MiB | Phase 6 |
| gateway | 52 MiB of 256 MiB limit | Phase 6 |
| prometheus | 27 MiB | Phase 6 |

## Serving latency (Phase 6, 5 requests/s for 60 s, 300 requests)
| Path | Errors | p50 | p95 |
| --- | --- | --- | --- |
| gateway -> blue slot `/v1/predict` | 0 | 27.9 ms | 35.8 ms |
| blue slot direct | 0 | n/a | 21.8 ms |
| gateway `/v1/explain` (Phase 7, XGBoost TreeSHAP) | 0 | 28.8 ms | 45.7 ms |

## Start-up times (wall clock)
| Step | Seconds |
| --- | --- |
| `up --wait postgres minio` (images cached) | 11 |
| `run --rm init` (pip install inside python:3.11-slim) | about 40 (not timed precisely) |
| MLflow image build + start | 83–93 (first build) |
| serving + gateway image build | 139 |
| `up --wait` of slots, gateway, prometheus (cached images) | 28 |
| `/admin/load` of v1 from MLflow/MinIO | 3.2 |

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
Airflow memory (Phase 9), Phase 11 re-measurement.

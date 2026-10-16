# Phase 6 evidence – single-model inference service

## Stack
`docker compose --profile core up -d --wait` brings up postgres, minio, mlflow, serving-blue, serving-green, gateway and prometheus; all healthy in 28 s (images cached). Images: `fairxai-serving:dev` and `fairxai-gateway:dev` (2.43 GB each; build 139 s). Ports bound to 127.0.0.1 only.

## Live checks (`data/runs/phase6_live.json`; keys read from `.env`, never printed)
| Check | Result |
| --- | --- |
| `POST blue/admin/load {"model_version":1}` | 200 `{loaded: true, model_version: 1, previous_version: null}`, 3.2 s |
| blue `/readyz`, gateway `/readyz` | 200 `{ready: true, model_version: 1}` |
| `GET gateway/v1/model` | v1, run `e29c476a`, threshold 0.53, slot blue, git commit and data_version_id present |
| `POST gateway/v1/predict` | 200, `prediction=1`, `probability_default=0.5459`, `X-Request-ID` present; row written to `predictions` (version 1, slot blue) |
| Wrong API key | 401 |
| `AGE: 7` | 422 `validation_error`, detail `features.AGE: Input should be greater than or equal to 18` |
| Unknown field | 422 |
| Rows added to `predictions` by the three rejected requests | **0** |
| `/v1/explain` | 501 (the only permitted Phase 6 stub; removed in Phase 7) |
| Flip to green with no model | 409 `slot_not_ready`; gateway stays on blue |

## Latency (N-12 target: predict p95 ≤ 200 ms at 5 requests/s)
Measured with `fairxai.deploy.probes.send` at a fixed 5 requests/s for 60 s (300 eval rows, `X-Probe: true` so nothing is logged):
| Path | Errors | p50 | p95 |
| --- | --- | --- | --- |
| Through gateway | 0 / 300 | 27.9 ms | **35.8 ms** |
| Direct to blue slot | 0 / 300 | — | 21.8 ms |

Prometheus-side p95 from `fairxai_request_latency_seconds` (5 m window): 21.9 ms. Measured on this dev machine with the host also running other containers. This is a measurement, not a guarantee.

## Prometheus
- Targets `serving/blue`, `serving/green`, `gateway`: all `up`.
- All metric names present: `fairxai_requests_total, fairxai_request_latency_seconds, fairxai_predictions_total, fairxai_model_loaded, fairxai_model_load_failures_total, fairxai_log_failures_total, gateway_active_slot, gateway_flips_total, gateway_upstream_errors_total, gateway_request_latency_seconds`.
- `promtool check rules`: SUCCESS, 5 rules. Before any model was loaded, `ActiveSlotWithoutModel` went `pending` as intended; it went `inactive` after the load.
- `High5xxRate` **fired** after the first live attempt, correctly: 600 predictions got 503 while the slot had no model (see incident below). It clears once those requests leave its 5-minute window.

## Container memory after traffic (`docker stats`)
serving-blue 207 MiB (model loaded) of 1 GiB limit; serving-green 75 MiB (empty); gateway 52 MiB of 256 MiB; prometheus 27 MiB; mlflow 1.69 GiB.

## Tests
`tests/api/test_serving.py` and `tests/api/test_gateway.py` (slots are real FastAPI apps with a synthetic fixture model; the gateway reaches them through an in-process ASGI transport): schema accept and reject at every field boundary (24 cases), missing and unknown fields, malformed JSON, 413, every status code, auth with right/wrong/missing/admin keys, client key on admin route = 403, keys absent from logs and bodies, `/readyz` 503 before load / after a failed load / on version mismatch, a failed `/admin/load` keeps the previous model serving, startup load of `EXPECTED_MODEL_VERSION`, probe requests not logged, `X-Logged: false` on log failure, feedback duplicates and unknown ids, 1–1000 label bounds, metric names after traffic, gateway forwarding, 502 on an unreachable slot, flip refused to a not-ready slot, **active slot survives a gateway restart** (real Postgres). Full suite after Phase 6: see the commit message.

## Incidents found and fixed
1. **MLflow 3 host-header check**: the server rejected `Host: mlflow:5000` from containers with 403 "Invalid Host header - possible DNS rebinding attack". Fixed with `--allowed-hosts` on the MLflow server (internal names and localhost only).
2. That 403 was reported by the slot as **404 "unknown model version"**. Fixed: only `RESOURCE_DOES_NOT_EXIST` maps to 404; any other registry error is a 500 and the slot keeps its previous model.

## Exit criteria
- [x] API tests pass.
- [x] p95 latency at 5 requests/s measured and recorded.
- [x] Prometheus shows all metric names.
- [x] A bad key and a malformed body are rejected and not logged as predictions.

## Not done / caveats
- `make` is not installed on this Windows host; the equivalent `docker compose` commands were run directly. The Makefile targets were updated but not executed through `make`.
- Image size (2.43 GB) is large because the serving lock includes SHAP and full scientific dependencies; not optimised.

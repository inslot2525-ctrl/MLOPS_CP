COMPOSE = docker compose --env-file .env -f infra/compose/docker-compose.yml
PY ?= python

.PHONY: train env up up-lite down ps health logs reset lint test test-integration

env:
	$(PY) scripts/make_env.py

# up-lite = core without Airflow (Airflow services arrive in Phase 9)
up-lite: env
	$(COMPOSE) --profile core up -d --wait postgres minio
	$(COMPOSE) --profile core run --rm init
	$(COMPOSE) --profile core up -d --build --wait mlflow serving-blue serving-green gateway prometheus

up: up-lite

down:
	$(COMPOSE) --profile core down

ps:
	$(COMPOSE) --profile core ps

health:
	$(COMPOSE) --profile core ps --format "{{.Service}} {{.Status}}"

logs:
	$(COMPOSE) --profile core logs --tail=100 $(SVC)

reset:
	$(COMPOSE) --profile core down -v
	$(PY) -c "import shutil,pathlib; [shutil.rmtree(p, ignore_errors=True) for p in ('data/ingest','data/retrain')]; pathlib.Path('.deployed_tag').unlink(missing_ok=True)"

lint:
	$(PY) -m ruff check .
	$(PY) -m ruff format --check .
	$(PY) scripts/check_tree.py

test:
	$(PY) -m pytest tests/unit tests/data tests/model tests/api

test-integration:
	$(PY) -m pytest tests/integration

train:
	$(PY) -m fairxai.models.train --track

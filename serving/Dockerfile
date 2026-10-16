# Build context: repository root.  docker build -f serving/Dockerfile .
FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONHASHSEED=0
WORKDIR /app
COPY requirements/serving.lock requirements/serving.lock
RUN pip install --no-cache-dir -r requirements/serving.lock
COPY pyproject.toml ./
COPY src ./src
RUN pip install --no-cache-dir --no-deps .
COPY configs ./configs
COPY serving ./serving
ENV FAIRXAI_CONFIG_DIR=/app/configs
RUN useradd --create-home --uid 10001 app
USER app
EXPOSE 8000
CMD ["uvicorn", "--factory", "serving.app.main:app_factory", "--host", "0.0.0.0", "--port", "8000"]

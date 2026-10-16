FROM python:3.11-slim
RUN pip install --no-cache-dir "mlflow==3.17.1" "psycopg2-binary" "boto3"
RUN useradd --create-home mlflow
USER mlflow
EXPOSE 5000

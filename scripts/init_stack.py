"""One-shot init: create Postgres databases, apply schema, create MinIO buckets. Idempotent."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import boto3
import psycopg
from botocore.client import Config
from botocore.exceptions import ClientError

DATABASES = ["airflow", "mlflow", "fairxai"]
BUCKETS = ["mlflow", "dvc"]


def init_postgres(host: str, user: str, password: str, schema_path: str) -> None:
    admin = f"postgresql://{user}:{password}@{host}:5432/postgres"
    with psycopg.connect(admin, autocommit=True) as conn:
        for db in DATABASES:
            exists = conn.execute("SELECT 1 FROM pg_database WHERE datname = %s", (db,)).fetchone()
            if not exists:
                conn.execute(f'CREATE DATABASE "{db}"')
                print(f"created database {db}")
    with psycopg.connect(f"postgresql://{user}:{password}@{host}:5432/fairxai") as conn:
        conn.execute(Path(schema_path).read_text(encoding="utf-8"))
        conn.commit()
    print("schema applied")


def init_minio(endpoint: str, key: str, secret: str) -> None:
    s3 = boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=key,
        aws_secret_access_key=secret,
        config=Config(signature_version="s3v4"),
        region_name="us-east-1",
    )
    for b in BUCKETS:
        try:
            s3.head_bucket(Bucket=b)
        except ClientError:
            s3.create_bucket(Bucket=b)
            print(f"created bucket {b}")


def main() -> int:
    init_postgres(
        os.environ.get("POSTGRES_HOST", "postgres"),
        os.environ["POSTGRES_USER"],
        os.environ["POSTGRES_PASSWORD"],
        os.environ.get("SCHEMA_PATH", "/work/infra/postgres/schema.sql"),
    )
    init_minio(
        os.environ["MLFLOW_S3_ENDPOINT_URL"], os.environ["MINIO_ROOT_USER"], os.environ["MINIO_ROOT_PASSWORD"]
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

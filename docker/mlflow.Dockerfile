FROM python:3.11-slim

RUN pip install --no-cache-dir mlflow==2.22.5 psycopg2-binary boto3 "sqlalchemy==2.0.36"

EXPOSE 5000
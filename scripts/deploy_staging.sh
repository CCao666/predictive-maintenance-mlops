#!/bin/sh

set -eu

WAIT_TIMEOUT="${DEPLOY_WAIT_TIMEOUT:-180}"

compose() {
  docker compose -f compose.yaml -f compose.staging.yaml "$@"
}

echo "Pulling application images tagged ${IMAGE_TAG:-main}..."
compose pull \
  mlflow inference-api stream-consumer prediction-writer stream-producer \
  drift-monitor retrain-coordinator alert-webhook

echo "Starting the model registry..."
compose up -d postgres minio minio-init mlflow
compose up -d --wait --wait-timeout "$WAIT_TIMEOUT" postgres minio mlflow

echo "Checking the champion model..."
compose run --rm --no-deps inference-api python -c \
  "from mlflow import MlflowClient; v = MlflowClient().get_model_version_by_alias('predictive-maintenance-rul', 'champion'); print(f'Found champion model version {v.version}')"

echo "Starting inference, streaming, monitoring, and alerting..."
compose up -d --wait --wait-timeout "$WAIT_TIMEOUT" \
  kafka inference-api drift-monitor alert-webhook alertmanager prometheus
compose up -d \
  grafana stream-consumer prediction-writer retrain-coordinator \
  stream-producer

echo "Deployment complete. API: http://127.0.0.1:8000/ready"
echo "Grafana: http://127.0.0.1:3000"

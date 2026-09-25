#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

for command_name in docker kind kubectl; do
  if ! command -v "$command_name" >/dev/null 2>&1; then
    echo "Missing required command: $command_name" >&2
    exit 1
  fi
done

docker info >/dev/null

if [[ ! -f .env ]]; then
  cp .env.example .env
  echo "Created .env from .env.example"
fi

echo "Starting PostgreSQL, MinIO, MLflow, and Kafka..."
docker compose up -d --build postgres minio minio-init mlflow kafka
docker compose up -d --wait --wait-timeout 180 postgres minio mlflow kafka

echo "Building the shared application image..."
docker build \
  -t predictive-maintenance-api:local \
  -t predictive-maintenance-api:latest \
  .

cat <<'EOF'

Bootstrap complete.

Before deploying to kind, train and register a champion model:
  export MLFLOW_TRACKING_URI=http://127.0.0.1:5050
  python -m training.tune_lstm --epochs 60
  python -m training.register_model

Then run:
  ./scripts/deploy_kind.sh
EOF

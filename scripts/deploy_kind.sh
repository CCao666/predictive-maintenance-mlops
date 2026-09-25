#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CLUSTER_NAME="${KIND_CLUSTER_NAME:-predictive-maintenance}"
NAMESPACE="predictive-maintenance"
METRICS_SERVER_URL="https://github.com/kubernetes-sigs/metrics-server/releases/latest/download/components.yaml"
cd "$ROOT_DIR"

if [[ ! -f .env ]]; then
  echo "Missing .env. Run ./scripts/bootstrap_demo.sh first." >&2
  exit 1
fi

set -a
# shellcheck disable=SC1091
source .env
set +a

: "${MINIO_ROOT_USER:?Set MINIO_ROOT_USER in .env}"
: "${MINIO_ROOT_PASSWORD:?Set MINIO_ROOT_PASSWORD in .env}"

echo "Checking the MLflow champion model..."
if ! docker compose run --rm --no-deps inference-api python -c \
  "from mlflow import MlflowClient; v = MlflowClient().get_model_version_by_alias('predictive-maintenance-rul', 'champion'); print(f'Champion model version: {v.version}')"; then
  echo "No champion model is registered. Train and register one before deploying." >&2
  exit 1
fi

docker compose stop inference-api stream-consumer >/dev/null 2>&1 || true
docker compose rm -f inference-api stream-consumer >/dev/null 2>&1 || true

if ! kind get clusters | grep -Fxq "$CLUSTER_NAME"; then
  kind create cluster --name "$CLUSTER_NAME" --wait 120s
fi

kind load docker-image predictive-maintenance-api:local --name "$CLUSTER_NAME"

kubectl apply -f "$METRICS_SERVER_URL"
if ! kubectl get deployment metrics-server -n kube-system -o json \
  | grep -q -- '--kubelet-insecure-tls'; then
  kubectl patch deployment metrics-server -n kube-system --type=json \
    -p='[{"op":"add","path":"/spec/template/spec/containers/0/args/-","value":"--kubelet-insecure-tls"}]'
fi
kubectl rollout status deployment/metrics-server -n kube-system --timeout=180s

kubectl create namespace "$NAMESPACE" --dry-run=client -o yaml \
  | kubectl apply -f -
kubectl create secret generic predictive-maintenance-secrets \
  --namespace "$NAMESPACE" \
  --from-literal=AWS_ACCESS_KEY_ID="$MINIO_ROOT_USER" \
  --from-literal=AWS_SECRET_ACCESS_KEY="$MINIO_ROOT_PASSWORD" \
  --dry-run=client -o yaml \
  | kubectl apply -f -

kubectl apply -k k8s
kubectl rollout restart deployment/inference-api -n "$NAMESPACE"
kubectl rollout status deployment/inference-api -n "$NAMESPACE" --timeout=600s
kubectl rollout status deployment/stream-consumer -n "$NAMESPACE" --timeout=300s

docker compose up -d --no-deps \
  prediction-writer drift-monitor retrain-coordinator \
  alert-webhook alertmanager prometheus grafana
docker compose up -d --no-deps stream-producer

echo
kubectl get pods,service,hpa -n "$NAMESPACE"
echo
echo "Deployment complete. Run ./scripts/verify_demo.sh"

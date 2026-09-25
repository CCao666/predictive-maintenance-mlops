#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NAMESPACE="predictive-maintenance"
cd "$ROOT_DIR"

echo "Checking Kubernetes rollouts..."
kubectl rollout status deployment/inference-api -n "$NAMESPACE" --timeout=300s
kubectl rollout status deployment/stream-consumer -n "$NAMESPACE" --timeout=300s

echo "Checking inference endpoints from inside the cluster..."
kubectl exec -n "$NAMESPACE" deployment/stream-consumer -- python -c \
  "import urllib.request; paths=('health','ready','model-info'); [print(f'/{p}: {urllib.request.urlopen(f\"http://inference-api:8000/{p}\", timeout=10).read().decode()}') for p in paths]"

echo "Checking the Kafka consumer..."
kubectl logs -n "$NAMESPACE" deployment/stream-consumer --tail=100 \
  | grep -q "Successfully joined group"

echo
kubectl get pods,service,hpa -n "$NAMESPACE"
echo
docker compose ps \
  postgres minio mlflow kafka prediction-writer drift-monitor \
  retrain-coordinator alert-webhook alertmanager prometheus grafana

echo
echo "Demo verification passed."
echo "Optional local API access:"
echo "  kubectl port-forward -n $NAMESPACE service/inference-api 8000:8000"

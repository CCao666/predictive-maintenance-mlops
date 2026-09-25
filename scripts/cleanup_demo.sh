#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CLUSTER_NAME="${KIND_CLUSTER_NAME:-predictive-maintenance}"
cd "$ROOT_DIR"

if kind get clusters | grep -Fxq "$CLUSTER_NAME"; then
  kind delete cluster --name "$CLUSTER_NAME"
fi

docker compose down --remove-orphans

echo "Demo stopped. Persistent Docker volumes were kept."
echo "Use 'docker compose down --volumes' only when you intend to erase local data."

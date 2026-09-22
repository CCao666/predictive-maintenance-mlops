# Real-Time Predictive Maintenance ML Platform

[![CI](https://github.com/CCao666/predictive-maintenance-mlops/actions/workflows/ci.yml/badge.svg)](https://github.com/CCao666/predictive-maintenance-mlops/actions/workflows/ci.yml)

Predicts remaining useful life (RUL) from NASA C-MAPSS turbofan sensor data.
This production-style simulation covers training and MLflow model registration,
FastAPI inference, Kafka sensor replay, PostgreSQL prediction storage,
Prometheus/Grafana monitoring, Alertmanager notifications, and CI/CD to GHCR.
The LSTM architecture follows `devwithmohit/predictive-maintenance-manufacturing-system`.

## Setup

Use Python 3.10 and Docker Compose v2 or newer.

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Model weights, local MLflow runs, and databases are excluded from Git. Application
tests and the isolated CI stack work without them. Copy the development settings
before starting the service stack:

```bash
cp .env.example .env
```

## Quick checks

```bash
source .venv/bin/activate
python -m pytest -q
python -m training.inspect_data
```

## Train and track

Start the shared tracking infrastructure, then point host-side training commands
at it:

```bash
docker compose up -d --build postgres minio minio-init mlflow
export MLFLOW_TRACKING_URI=http://127.0.0.1:5050
python -m training.train_lstm --run-name reference-lstm
```

Training uses engine-level splitting, a train-only `RobustScaler`, 50-cycle
sequences, capped RUL labels, early stopping, and learning-rate reduction.

## MLflow

The Compose stack stores MLflow metadata in its own PostgreSQL database and
stores model artifacts in the `mlflow` MinIO bucket. MLflow proxies artifact
traffic, so training and inference clients need only the tracking URL; they do
not receive MinIO credentials.

To select and register a new champion:

```bash
export MLFLOW_TRACKING_URI=http://127.0.0.1:5050
python -m training.tune_lstm
python -m training.register_model
```

To copy the champion from this project's earlier local SQLite Registry instead
of retraining it:

```bash
python -m training.migrate_registry
```

The migration is idempotent: rerunning it reuses the version with the same
source run and restores its `champion` alias.

The registered production candidate is available as:

```text
models:/predictive-maintenance-rul@champion
```

Run the official FD001 holdout evaluation and attach the results to the
champion's MLflow run:

```bash
MLFLOW_TRACKING_URI=http://127.0.0.1:5050 \
  python -m training.evaluate_test --log-to-mlflow
```

This evaluates one final prediction per test engine, using `RUL_FD001.txt` as
ground truth and the upstream inference convention of left zero-padding short
sequences before scaling. Results are saved under `artifacts/test_evaluation/`.

## Inference API

```bash
MLFLOW_TRACKING_URI=http://127.0.0.1:5050 \
  uvicorn inference.main:app --host 0.0.0.0 --port 8000
```

Useful endpoints:

```text
GET  /health
GET  /ready
GET  /model-info
POST /predict
GET  /metrics
```

Interactive API documentation is available at `http://127.0.0.1:8000/docs`.

## Docker

On a new machine, initialize the Registry and register a champion before
starting the full stack:

```bash
cp .env.example .env
docker compose up -d --build postgres minio minio-init mlflow

# Choose one:
python -m training.migrate_registry  # when the old local Registry is available
# or train and register using the commands in "Train and track"

docker compose up -d --build
```

Then verify the container and loaded champion model:

```bash
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/ready
curl http://127.0.0.1:8000/model-info
```

Stop it with:

```bash
docker compose down
```

Development interfaces:

- MLflow UI: `http://127.0.0.1:5050`
- MinIO API: `http://127.0.0.1:9000`
- MinIO Console: `http://127.0.0.1:9001`

The default MinIO credentials live in `.env.example` for local development.
Copy them into an untracked `.env` and replace them before using the stack on a
shared network. PostgreSQL and MinIO data persist in Docker named volumes, so
`docker compose down` keeps Registry state and `docker compose down -v` removes
it intentionally.

## Kafka sensor stream

The Compose stack also starts Kafka, replays `test_FD001.txt`, keeps an
independent 50-cycle window for every engine, and sends each ready window to the
inference API. Successful predictions are published to
`engine-rul-predictions` and saved in PostgreSQL:

```bash
docker compose up -d --build
docker compose logs -f stream-consumer
```

Kafka is available to local tools at `localhost:9092`. Inside Compose, services
connect to `kafka:19092`. The producer exits after publishing the dataset; the
broker, inference API, and consumer keep running.

Inspect saved predictions:

```bash
docker compose exec postgres psql -U maintenance -d maintenance -c \
  "SELECT engine_id, time_cycle, predicted_rul, alert_level, predicted_at FROM rul_predictions ORDER BY predicted_at DESC LIMIT 20;"
```

The prediction table is idempotent by engine, cycle, and model version, so a
dataset replay updates the same logical prediction rather than creating a
duplicate.

To replay a smaller sample from the host:

```bash
python -m streaming.producer --limit 200 --delay 0.05
```

Stop and remove the stack with `docker compose down`.

## Monitoring

Prometheus scrapes inference metrics every two seconds and keeps seven days of
local history. Grafana starts with the Prometheus data source and Predictive
Maintenance dashboard already provisioned:

- Prometheus: `http://127.0.0.1:9090`
- Grafana: `http://127.0.0.1:3000`
- Alertmanager: `http://127.0.0.1:9093`
- Dashboard: **Predictive Maintenance / Predictive Maintenance Overview**

The dashboard includes a per-engine selector, RUL history, latest RUL,
prediction throughput, p95 latency, and errors. Local anonymous read access is
enabled for convenience; the development administrator credentials are
`admin` / `admin` and must be changed before any shared deployment.

## Alerts and operating actions

Prometheus evaluates the latest fresh prediction for every engine. A prediction
older than five minutes is excluded from RUL alerts. A separate stale-data alert
fires when the entire prediction stream has been silent for five minutes.
Alertmanager groups repeated notifications and sends firing
and resolved states to the local webhook, which stores them in PostgreSQL.

| Predicted RUL | Level | Suggested action |
| --- | --- | --- |
| `<= 15` | Urgent | Inspect immediately; assess derating or safe shutdown. |
| `15 < RUL <= 40` | Critical | Schedule maintenance at the earliest safe opportunity. |
| `40 < RUL <= 50` | Warning | Create a work order and increase inspection frequency. |
| `50 < RUL <= 60` | Watch | Review the trend and maintenance capacity. |
| `> 60` | Healthy | Continue normal monitoring. |

These thresholds are conservative operational policy, not proof that an engine
is safe. They should be calibrated with failure cost, inspection capacity, and
real fleet data before deployment. The stack also alerts on inference downtime,
prediction errors, high p95 latency, and stale or missing prediction data.

Inspect persisted alert history:

```bash
docker compose exec postgres psql -U maintenance -d maintenance -c \
  "SELECT status, severity, alert_name, engine_id, summary, last_received_at FROM alerts ORDER BY last_received_at DESC;"
```

For this local simulation the webhook is the notification receiver. A real
deployment would add an authenticated on-call receiver such as PagerDuty, Slack,
or email without changing the Prometheus rules.

Prometheus and Grafana history is stored in named Docker volumes. Use
`docker compose down -v` only when you intentionally want to delete it.

`upstream-reference/` is read-only reference material and is not part of the
project implementation.

## CI/CD

GitHub Actions runs on pushes, pull requests, and manual dispatch. It runs the
application tests, validates Compose and monitoring rules, builds the service
image, and tests two complete paths in an isolated Compose project:

- Sensor readings → Kafka → consumer → FastAPI → result topic → PostgreSQL,
  including duplicate-result replay to verify idempotent writes.
- Low-RUL prediction → Prometheus rule → Alertmanager → webhook → PostgreSQL,
  followed by a healthy prediction to verify the same alert becomes resolved.

Only the model manager is substituted with a deterministic test implementation.
Kafka, HTTP endpoints, PostgreSQL, Prometheus rules, and Alertmanager routing are
real. These tests check service integration, not model accuracy. The CI image is
built and exercised on the runner. Reports and service logs are retained as
Actions artifacts for seven days.

After every successful push to `main`, the CD job publishes two deployable
images to GitHub Container Registry:

- `ghcr.io/ccao666/predictive-maintenance-api`
- `ghcr.io/ccao666/predictive-maintenance-mlflow`

Each image receives `main`, `latest`, and immutable `sha-<commit>` tags. A `v*`
Git tag also produces a matching release image tag. Images support both
`linux/amd64` production hosts and `linux/arm64` development machines. Pull
requests only run CI and never publish images.

Deploy a tested image set to a Docker Compose staging host with:

```bash
cp .env.example .env
IMAGE_TAG=sha-<commit> sh scripts/deploy_staging.sh
```

The deployment uses `compose.staging.yaml` to pull published images instead of
building source on the host. It starts the Registry first and refuses to start
the inference stack unless `models:/predictive-maintenance-rul@champion` exists.
Persistent MLflow, MinIO, PostgreSQL, Prometheus, Grafana, and Alertmanager data
remain in named volumes when application containers are replaced.

To create a named release after the `main` workflow is green:

```bash
git tag v1.0.0
git push origin v1.0.0
```

This is continuous delivery: GitHub produces versioned, deployable artifacts,
while deployment to a particular host is an explicit operation. A future cloud
or self-hosted runner can call the same script after adding that environment's
credentials and approval rules.

Run the same integration checks locally (requires Docker):

```bash
docker build -t predictive-maintenance-ci:latest .
docker compose -f compose.ci.yaml up -d --wait --wait-timeout 180 postgres kafka inference-api alert-webhook alertmanager prometheus
docker compose -f compose.ci.yaml up -d stream-consumer prediction-writer
RUN_INTEGRATION_TESTS=1 python -m pytest tests/integration -v
docker compose -f compose.ci.yaml down --volumes --remove-orphans
```

The test project uses its own database and broker and host ports 15432, 19092,
and 18000. The cleanup command above removes only this disposable test stack.
Ordinary `python -m pytest` skips its two integration tests unless
`RUN_INTEGRATION_TESTS=1` is set.

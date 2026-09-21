# Real-Time Predictive Maintenance ML Platform

[![CI](https://github.com/CCao666/predictive-maintenance-mlops/actions/workflows/ci.yml/badge.svg)](https://github.com/CCao666/predictive-maintenance-mlops/actions/workflows/ci.yml)

Predicts remaining useful life (RUL) from NASA C-MAPSS turbofan sensor data.
This local production simulation covers training and MLflow model registration,
FastAPI inference, Kafka sensor replay, PostgreSQL prediction storage,
Prometheus/Grafana monitoring, Alertmanager notifications, and automated CI.
The LSTM architecture follows `devwithmohit/predictive-maintenance-manufacturing-system`.

## Setup

Use Python 3.10 and Docker Compose v2 or newer.

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Model weights, local MLflow runs, and databases are excluded from Git. Application
tests and the isolated CI stack work without them. To create your own champion:

```bash
python -m training.tune_lstm
python -m training.register_model
```

The demo Compose file still reflects the original machine's MLflow artifact
mount. On a new machine, update that mount to match the absolute artifact path
in your newly created MLflow runs before starting the model-backed demo.

## Quick checks

```bash
source .venv/bin/activate
python -m pytest -q
python -m training.inspect_data
```

## Train and track

```bash
python -m training.train_lstm --run-name reference-lstm
```

Training uses engine-level splitting, a train-only `RobustScaler`, 50-cycle
sequences, capped RUL labels, early stopping, and learning-rate reduction.

## MLflow

```bash
mlflow server \
  --backend-store-uri sqlite:///mlflow.db \
  --host 127.0.0.1 \
  --port 5000
```

The registered production candidate is available as:

```text
models:/predictive-maintenance-rul@champion
```

Run the official FD001 holdout evaluation and attach the results to the
champion's MLflow run:

```bash
python -m training.evaluate_test --log-to-mlflow
```

This evaluates one final prediction per test engine, using `RUL_FD001.txt` as
ground truth and the upstream inference convention of left zero-padding short
sequences before scaling. Results are saved under `artifacts/test_evaluation/`.

## Inference API

```bash
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

Build and start the inference service with its local MLflow Registry:

```bash
docker compose up --build
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

The Compose file mounts `mlflow.db` read-only and mounts `mlruns/` for artifact
access. MLflow needs write access to the artifact directory to create its small
registered-model metadata file. The unusual absolute `mlruns` target is required
because existing local runs record their artifact locations as absolute paths.
A later production deployment should use a separate MLflow server and shared
artifact storage instead.

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

## Continuous integration

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
built and exercised on the runner; automatic deployment and registry publishing
are not configured. Reports and service logs are retained as Actions artifacts
for seven days.

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

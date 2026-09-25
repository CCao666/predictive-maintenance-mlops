# Real-Time Predictive Maintenance ML Platform

[![CI](https://github.com/CCao666/predictive-maintenance-mlops/actions/workflows/ci.yml/badge.svg)](https://github.com/CCao666/predictive-maintenance-mlops/actions/workflows/ci.yml)

Predicts remaining useful life (RUL) from NASA C-MAPSS turbofan sensor data.
This production-style simulation covers training and MLflow model registration,
FastAPI inference, Kafka sensor replay, PostgreSQL prediction storage,
Prometheus/Grafana monitoring, Alertmanager notifications, and CI/CD to GHCR.
The LSTM architecture follows `devwithmohit/predictive-maintenance-manufacturing-system`.

## Demo

The Grafana dashboard follows per-engine RUL predictions and inference health in
real time. The example below replays the FD001 test split through Kafka and the
deployed inference pipeline.

![FD001 per-engine RUL predictions](docs/assets/grafana-fd001-rul.png)

The online drift monitor uses FD001 as its reference distribution. Replaying
FD001 settles back to no detected drift, while FD002's additional operating
conditions produce a critical shift across 22 sensor and operating features.

| FD001 reference replay | FD002 shifted replay |
| --- | --- |
| ![FD001 drift baseline](docs/assets/grafana-fd001-drift-baseline.png) | ![FD002 critical data drift](docs/assets/grafana-fd002-drift.png) |

The production candidate is versioned in the MLflow Model Registry and loaded
by alias, so inference uses `models:/predictive-maintenance-rul@champion`
instead of a hard-coded model version.

![MLflow registered champion model](docs/assets/mlflow-champion-model.png)

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

## Kubernetes (local kind cluster)

Only the stateless `inference-api` and `stream-consumer` workloads run in
Kubernetes. PostgreSQL, Kafka, MLflow, MinIO, Prometheus, Grafana, Alertmanager,
and the prediction writer remain in Compose. Docker Desktop, `kind`, and
`kubectl` are required.

Prepare the Compose dependencies and application image:

```bash
./scripts/bootstrap_demo.sh
```

On a fresh clone, train and register the champion model before deployment:

```bash
export MLFLOW_TRACKING_URI=http://127.0.0.1:5050
python -m training.train_lstm --run-name reference-lstm
python -m training.register_model
```

Deploy and verify the hybrid kind + Compose environment:

```bash
./scripts/deploy_kind.sh
./scripts/verify_demo.sh
```

The deployment script creates or reuses the kind cluster, installs Metrics
Server, injects the local MinIO credentials as a Kubernetes Secret, applies the
Kustomize resources, and waits for both Deployments. It also bridges MLflow's
Compose-internal `minio` artifact hostname through an ExternalName Service.

For local API access, keep this command running in a separate terminal:

```bash
kubectl port-forward -n predictive-maintenance service/inference-api 8000:8000
```

Stop the demo without deleting persistent Docker volumes:

```bash
./scripts/cleanup_demo.sh
```

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
  "SELECT dataset_id, engine_id, time_cycle, predicted_rul, alert_level, predicted_at FROM rul_predictions ORDER BY predicted_at DESC LIMIT 20;"
```

The prediction table is idempotent by dataset, engine, cycle, and model version,
so a dataset replay updates the same logical prediction rather than creating a
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

### Offline drift benchmark

Use FD001 training data as the reference distribution and compare all four
C-MAPSS test datasets:

```bash
python -m monitoring.evaluate_drift
```

The detector runs a two-sample Kolmogorov-Smirnov test on the 3 operating
settings and 21 sensor features. A feature is drifted when `p < 0.05` and the
KS statistic is at least `0.20`; requiring both prevents large datasets from
turning negligible differences into alerts. Reports are written to
`artifacts/drift/`, including a compact `summary.json` and one detailed report
per dataset.

FD001 is the no-drift control. FD002 introduces multiple operating conditions,
FD003 introduces an additional fault mode, and FD004 contains both changes.

### Online drift monitoring

`drift-monitor` independently consumes every sensor event from Kafka. It keeps
a bounded rolling window per `dataset_id`, compares that window with
`train_FD001.txt`, publishes each result to `model-drift-events`, and exposes
Prometheus metrics on `http://127.0.0.1:8081/metrics`. The production defaults
wait for 1,000 observations, retain 5,000, and recheck every 500 observations.

Replay a shifted dataset and watch the result:

```bash
docker compose up -d --build drift-monitor retrain-coordinator prometheus grafana
docker compose run --rm stream-producer \
  python -m streaming.producer --data data/cmapss/test_FD002.txt \
  --dataset-id FD002 --limit 2500 --delay 0
docker compose logs -f drift-monitor retrain-coordinator
```

Grafana adds a dataset selector and panels for KS score, drifted-feature ratio,
severity, drifted-feature count, and monitor throughput. Prometheus sends
`DataDriftWarning`, `DataDriftCritical`, `DriftMonitorDown`, and
`DriftChecksStale` alerts to Alertmanager.

### Guarded retraining

Every drift event is persisted in `drift_reports`. Three consecutive critical
checks create one `pending_review` request, with a 24-hour cooldown and no
duplicate active request. Detection never starts training or replaces the
production model automatically.

Inspect the audit trail:

```bash
docker compose exec postgres psql -U maintenance -d maintenance -c \
  "SELECT dataset_id, severity, drift_score, drifted_feature_ratio, observed_at FROM drift_reports ORDER BY observed_at DESC LIMIT 20;"
docker compose exec postgres psql -U maintenance -d maintenance -c \
  "SELECT id, dataset_id, trigger, status, requested_at FROM retraining_requests ORDER BY requested_at DESC;"
```

An operator can approve and train a request from the host:

```bash
export DATABASE_URL=postgresql://maintenance:maintenance@127.0.0.1:5432/maintenance
export MLFLOW_TRACKING_URI=http://127.0.0.1:5050
python -m training.retrain_candidate approve <request-id>
python -m training.retrain_candidate train <request-id> --epochs 60
```

The candidate combines FD001 training data with the affected dataset, is logged
to MLflow, and is compared with the current champion on both the affected
dataset and the original FD001 official holdouts. It becomes the `challenger`
only when all safety gates pass:

- On the affected dataset, RMSE improves by at least 5%, NASA score does not
  worsen, and severe lifetime overestimates do not increase.
- On FD001, RMSE degradation is at most 3%, NASA score does not worsen, and
  severe lifetime overestimates do not increase.

Even then, production promotion remains explicit:

```bash
python -m training.retrain_candidate promote <request-id>
```

This changes the MLflow `champion` alias only for a promotion-eligible request.

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
application tests, validates Compose, Kubernetes manifests, and monitoring
rules, builds the service image, and tests three complete paths in an isolated
Compose project. Kubernetes resources are rendered with Kustomize and checked
against their schemas with kubeconform before any image is published. A
lightweight kind smoke test also verifies Deployment rollout, health probes,
ClusterIP routing, and automatic Pod replacement without training a model:

- Sensor readings → Kafka → consumer → FastAPI → result topic → PostgreSQL,
  including duplicate-result replay to verify idempotent writes.
- Low-RUL prediction → Prometheus rule → Alertmanager → webhook → PostgreSQL,
  followed by a healthy prediction to verify the same alert becomes resolved.
- Persistent feature drift → Kafka drift event → Prometheus/Alertmanager and
  PostgreSQL audit records → guarded retraining request.

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
docker compose -f compose.ci.yaml up -d --wait --wait-timeout 180 postgres kafka inference-api drift-monitor alert-webhook alertmanager prometheus
docker compose -f compose.ci.yaml up -d stream-consumer prediction-writer retrain-coordinator
RUN_INTEGRATION_TESTS=1 python -m pytest tests/integration -v
docker compose -f compose.ci.yaml down --volumes --remove-orphans
```

The test project uses its own database and broker and host ports 15432, 19092,
and 18000. The cleanup command above removes only this disposable test stack.
Ordinary `python -m pytest` skips its three integration tests unless
`RUN_INTEGRATION_TESTS=1` is set.

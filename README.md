# Real-Time Predictive Maintenance ML Platform

Predicts remaining useful life (RUL) from NASA C-MAPSS turbofan sensor data.
The current implementation covers data preparation, a Random Forest baseline,
an LSTM reference model, MLflow experiment tracking, parameter selection, and
model registration.

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

`upstream-reference/` is read-only reference material and is not part of the
project implementation.

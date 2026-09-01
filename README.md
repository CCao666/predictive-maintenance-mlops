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

`upstream-reference/` is read-only reference material and is not part of the
project implementation.

"""Prometheus metrics exposed by the inference API."""

from prometheus_client import Counter, Gauge, Histogram


PREDICTION_REQUESTS = Counter(
    "prediction_requests_total",
    "Total prediction requests",
)
PREDICTION_ERRORS = Counter(
    "prediction_errors_total",
    "Failed prediction requests",
)
PREDICTION_LATENCY = Histogram(
    "prediction_latency_seconds",
    "Prediction request latency",
)
PREDICTED_RUL = Histogram(
    "predicted_rul",
    "Distribution of predicted RUL values",
)
ENGINE_RUL = Gauge(
    "predicted_rul_by_engine",
    "Latest predicted RUL for each engine",
    ["engine_id"],
)
ENGINE_LAST_PREDICTION = Gauge(
    "engine_last_prediction_timestamp_seconds",
    "Unix timestamp of the latest prediction for each engine",
    ["engine_id"],
)
MODEL_INFO = Gauge(
    "model_info",
    "Currently loaded model",
    ["name", "version", "alias"],
)

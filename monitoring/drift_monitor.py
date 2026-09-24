"""Monitor Kafka sensor windows for C-MAPSS feature drift."""

import json
import logging
import os
from collections import defaultdict, deque
from pathlib import Path

import pandas as pd
from kafka import KafkaConsumer, KafkaProducer
from prometheus_client import Counter, Gauge, start_http_server
from pydantic import ValidationError

from monitoring.drift import detect_data_drift
from streaming.producer import DEFAULT_TOPIC
from streaming.schemas import DRIFT_TOPIC, DriftEvent, SensorEvent
from training.dataset import FEATURE_COLUMNS, load_cmapss


logger = logging.getLogger(__name__)
MESSAGES = Counter(
    "drift_monitor_messages_total",
    "Sensor messages processed by the drift monitor",
    ["dataset_id"],
)
CHECKS = Counter(
    "data_drift_checks_total",
    "Completed data drift checks",
    ["dataset_id", "severity"],
)
DRIFT_SCORE = Gauge(
    "data_drift_ks_score",
    "Maximum KS statistic in the latest drift check",
    ["dataset_id"],
)
DRIFT_RATIO = Gauge(
    "data_drift_feature_ratio",
    "Fraction of features drifted in the latest check",
    ["dataset_id"],
)
DRIFT_FEATURES = Gauge(
    "data_drift_features",
    "Number of drifted features in the latest check",
    ["dataset_id"],
)
DRIFT_SEVERITY = Gauge(
    "data_drift_severity",
    "Latest drift severity: 0 none, 1 warning, 2 critical",
    ["dataset_id"],
)
LAST_CHECK = Gauge(
    "data_drift_last_check_timestamp_seconds",
    "Unix timestamp of the latest drift check",
    ["dataset_id"],
)


class RollingDriftMonitor:
    """Keep bounded per-dataset windows and periodically check drift."""

    def __init__(
        self,
        reference: pd.DataFrame,
        window_size: int = 5_000,
        check_interval: int = 500,
        min_samples: int = 1_000,
    ) -> None:
        if min_samples < 30:
            raise ValueError("min_samples must be at least 30")
        if window_size < min_samples:
            raise ValueError("window_size must be at least min_samples")
        if check_interval <= 0:
            raise ValueError("check_interval must be greater than zero")

        self.reference = reference
        self.window_size = window_size
        self.check_interval = check_interval
        self.min_samples = min_samples
        self._windows: dict[str, deque[dict]] = defaultdict(
            lambda: deque(maxlen=window_size)
        )
        self._counts: dict[str, int] = defaultdict(int)

    def add(self, event: SensorEvent) -> dict | None:
        dataset_id = event.dataset_id
        self._windows[dataset_id].append(event.features.model_dump())
        self._counts[dataset_id] += 1

        sample_count = len(self._windows[dataset_id])
        if sample_count < self.min_samples:
            return None
        if self._counts[dataset_id] % self.check_interval:
            return None

        current = pd.DataFrame(
            self._windows[dataset_id],
            columns=FEATURE_COLUMNS,
        )
        return detect_data_drift(
            self.reference,
            current,
            min_samples=self.min_samples,
        )


def build_drift_event(dataset_id: str, report: dict) -> DriftEvent:
    return DriftEvent(
        dataset_id=dataset_id,
        severity=report["severity"],
        drift_detected=report["drift_detected"],
        drift_score=report["overall_drift_score"],
        drifted_feature_ratio=report["drifted_feature_ratio"],
        drifted_features=report["drifted_features"],
        reference_rows=report["reference_rows"],
        current_rows=report["current_rows"],
    )


def update_metrics(event: DriftEvent) -> None:
    severity_value = {"none": 0, "warning": 1, "critical": 2}
    labels = {"dataset_id": event.dataset_id}
    CHECKS.labels(event.dataset_id, event.severity).inc()
    DRIFT_SCORE.labels(**labels).set(event.drift_score)
    DRIFT_RATIO.labels(**labels).set(event.drifted_feature_ratio)
    DRIFT_FEATURES.labels(**labels).set(len(event.drifted_features))
    DRIFT_SEVERITY.labels(**labels).set(severity_value[event.severity])
    LAST_CHECK.labels(**labels).set(event.observed_at.timestamp())


def run() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    bootstrap_servers = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
    sensor_topic = os.getenv("KAFKA_TOPIC", DEFAULT_TOPIC)
    drift_topic = os.getenv("KAFKA_DRIFT_TOPIC", DRIFT_TOPIC)
    reference_path = Path(
        os.getenv("DRIFT_REFERENCE_PATH", "data/cmapss/train_FD001.txt")
    )
    metrics_port = int(os.getenv("DRIFT_METRICS_PORT", "8081"))

    monitor = RollingDriftMonitor(
        load_cmapss(reference_path),
        window_size=int(os.getenv("DRIFT_WINDOW_SIZE", "5000")),
        check_interval=int(os.getenv("DRIFT_CHECK_INTERVAL", "500")),
        min_samples=int(os.getenv("DRIFT_MIN_SAMPLES", "1000")),
    )
    consumer = KafkaConsumer(
        sensor_topic,
        bootstrap_servers=bootstrap_servers,
        group_id="data-drift-monitor",
        auto_offset_reset="earliest",
        enable_auto_commit=False,
        value_deserializer=lambda value: json.loads(value.decode()),
    )
    producer = KafkaProducer(
        bootstrap_servers=bootstrap_servers,
        key_serializer=lambda key: key.encode(),
        value_serializer=lambda value: json.dumps(value, default=str).encode(),
        acks="all",
    )
    start_http_server(metrics_port)
    logger.info("Drift metrics available on port %s", metrics_port)

    try:
        for message in consumer:
            try:
                sensor_event = SensorEvent.model_validate(message.value)
                MESSAGES.labels(sensor_event.dataset_id).inc()
                report = monitor.add(sensor_event)
                if report is not None:
                    event = build_drift_event(sensor_event.dataset_id, report)
                    update_metrics(event)
                    producer.send(
                        drift_topic,
                        key=event.dataset_id,
                        value=event.model_dump(),
                    ).get(timeout=10)
                    logger.info(
                        "dataset=%s severity=%s score=%.3f drifted=%s/24",
                        event.dataset_id,
                        event.severity,
                        event.drift_score,
                        len(event.drifted_features),
                    )
                consumer.commit()
            except (ValidationError, ValueError):
                logger.exception("Skipping invalid event in drift monitor")
                consumer.commit()
    finally:
        producer.close()
        consumer.close()


if __name__ == "__main__":
    run()

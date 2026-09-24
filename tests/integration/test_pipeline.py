"""Real Kafka, HTTP, PostgreSQL and Prometheus/Alertmanager integration tests."""

import json
import os
import time
from uuid import uuid4

import psycopg
import pytest
from kafka import KafkaProducer

from streaming.schemas import SensorEvent
from training.dataset import FEATURE_COLUMNS


pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        os.getenv("RUN_INTEGRATION_TESTS") != "1",
        reason="Start compose.ci.yaml and set RUN_INTEGRATION_TESTS=1",
    ),
]
DATABASE_URL = "postgresql://ci:ci@localhost:15432/maintenance_ci"


@pytest.fixture
def database():
    with psycopg.connect(DATABASE_URL, autocommit=True) as connection:
        yield connection


@pytest.fixture
def producer():
    producer = KafkaProducer(
        bootstrap_servers="localhost:19092",
        key_serializer=lambda value: str(value).encode(),
        value_serializer=lambda value: json.dumps(value).encode(),
        acks="all",
    )
    yield producer
    producer.close()


def send_readings(producer, engine_id, cycles, rul, dataset_id="FD001"):
    for cycle in cycles:
        event = SensorEvent(
            dataset_id=dataset_id,
            engine_id=engine_id,
            time_cycle=cycle,
            features={name: float(rul) for name in FEATURE_COLUMNS},
        )
        producer.send(
            "engine-sensor-readings", key=engine_id, value=event.model_dump()
        ).get(timeout=15)


def wait_for_row(database, query, parameters, timeout=90):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        row = database.execute(query, parameters).fetchone()
        if row is not None:
            return row
        time.sleep(1)
    pytest.fail(f"Timed out waiting for database row: {query}, {parameters}")


def test_sensor_stream_to_prediction_record(database, producer):
    engine = uuid4().int % 1_000_000_000 + 1
    send_readings(producer, engine, range(1, 51), rul=80)
    row = wait_for_row(
        database,
        "SELECT predicted_rul, alert_level, model_name, model_version "
        "FROM rul_predictions WHERE engine_id = %s AND time_cycle = 50",
        (engine,),
    )
    assert row == (80.0, "none", "ci-model", "ci-1")

    # Replay the actual serialized result event to test writer idempotence.
    saved = database.execute(
        "SELECT engine_id, time_cycle, predicted_rul, health_status, alert_level, "
        "model_name, model_version, model_alias, predicted_at "
        "FROM rul_predictions WHERE engine_id = %s", (engine,),
    )
    event = dict(zip([column.name for column in saved.description], saved.fetchone()))
    event["predicted_at"] = event["predicted_at"].isoformat()
    event["model_alias"] = "replay-check"
    producer.send("engine-rul-predictions", key=engine, value=event).get(timeout=15)
    wait_for_row(
        database,
        "SELECT id FROM rul_predictions WHERE engine_id = %s AND model_alias = %s",
        (engine, "replay-check"),
    )
    assert database.execute(
        "SELECT COUNT(*) FROM rul_predictions WHERE engine_id = %s", (engine,)
    ).fetchone()[0] == 1


def test_low_rul_alert_fires_and_resolves(database, producer):
    engine = uuid4().int % 1_000_000_000 + 1
    send_readings(producer, engine, range(1, 51), rul=10)
    alert_query = (
        "SELECT fingerprint, severity FROM alerts "
        "WHERE engine_id = %s AND alert_name = 'EngineRULUrgent' AND status = %s"
    )
    firing = wait_for_row(database, alert_query, (str(engine), "firing"))
    assert firing[1] == "critical"

    # The next real sensor event raises the prediction above every threshold.
    send_readings(producer, engine, [51], rul=80)
    resolved = wait_for_row(database, alert_query, (str(engine), "resolved"))
    assert resolved[0] == firing[0]


def test_persistent_drift_creates_retraining_request_and_alert(
    database,
    producer,
):
    engine = uuid4().int % 1_000_000_000 + 1
    send_readings(
        producer,
        engine,
        range(1, 91),
        rul=80,
        dataset_id="FD002",
    )

    request = wait_for_row(
        database,
        "SELECT trigger, status FROM retraining_requests "
        "WHERE dataset_id = %s",
        ("FD002",),
    )
    assert request == ("persistent_critical_data_drift", "pending_review")
    assert database.execute(
        "SELECT COUNT(*) FROM drift_reports WHERE dataset_id = %s",
        ("FD002",),
    ).fetchone()[0] >= 3

    alert = wait_for_row(
        database,
        "SELECT severity FROM alerts "
        "WHERE alert_name = 'DataDriftCritical' "
        "AND labels->>'dataset_id' = %s AND status = 'firing'",
        ("FD002",),
    )
    assert alert[0] == "critical"

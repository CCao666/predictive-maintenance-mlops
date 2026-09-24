"""Persist drift events and create guarded model retraining requests."""

import json
import logging
import os
from datetime import datetime, timedelta, timezone

from kafka import KafkaConsumer
from psycopg.types.json import Jsonb
from pydantic import ValidationError

from streaming.database import connect_with_retry
from streaming.schemas import DRIFT_TOPIC, DriftEvent


logger = logging.getLogger(__name__)


def should_request_retraining(
    recent_severities: list[str],
    last_request_at: datetime | None,
    *,
    consecutive_critical: int = 3,
    cooldown: timedelta = timedelta(hours=24),
    now: datetime | None = None,
) -> bool:
    """Require persistent critical drift and enforce a request cooldown."""
    if consecutive_critical <= 0:
        raise ValueError("consecutive_critical must be greater than zero")
    if len(recent_severities) < consecutive_critical:
        return False
    if any(
        severity != "critical"
        for severity in recent_severities[:consecutive_critical]
    ):
        return False

    now = now or datetime.now(timezone.utc)
    return last_request_at is None or now - last_request_at >= cooldown


def ensure_schema(connection) -> None:
    with connection.cursor() as cursor:
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS drift_reports (
                id BIGSERIAL PRIMARY KEY,
                dataset_id TEXT NOT NULL,
                severity TEXT NOT NULL,
                drift_score DOUBLE PRECISION NOT NULL,
                drifted_feature_ratio DOUBLE PRECISION NOT NULL,
                drifted_features JSONB NOT NULL,
                reference_rows INTEGER NOT NULL,
                current_rows INTEGER NOT NULL,
                observed_at TIMESTAMPTZ NOT NULL,
                received_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )
            """
        )
        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS drift_reports_dataset_time_idx
                ON drift_reports (dataset_id, observed_at DESC)
            """
        )
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS retraining_requests (
                id BIGSERIAL PRIMARY KEY,
                dataset_id TEXT NOT NULL,
                trigger TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'pending_review',
                drift_report_id BIGINT REFERENCES drift_reports(id),
                requested_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                notes TEXT
            )
            """
        )
        cursor.execute(
            "ALTER TABLE retraining_requests "
            "ADD COLUMN IF NOT EXISTS candidate_run_id TEXT"
        )
        cursor.execute(
            "ALTER TABLE retraining_requests "
            "ADD COLUMN IF NOT EXISTS candidate_version TEXT"
        )
        cursor.execute(
            "ALTER TABLE retraining_requests "
            "ADD COLUMN IF NOT EXISTS comparison JSONB"
        )
        cursor.execute(
            "ALTER TABLE retraining_requests "
            "ADD COLUMN IF NOT EXISTS completed_at TIMESTAMPTZ"
        )
        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS retraining_requests_dataset_time_idx
                ON retraining_requests (dataset_id, requested_at DESC)
            """
        )
    connection.commit()


def save_drift_event(connection, event: DriftEvent) -> int:
    with connection.cursor() as cursor:
        row = cursor.execute(
            """
            INSERT INTO drift_reports (
                dataset_id, severity, drift_score, drifted_feature_ratio,
                drifted_features, reference_rows, current_rows, observed_at
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING id
            """,
            (
                event.dataset_id,
                event.severity,
                event.drift_score,
                event.drifted_feature_ratio,
                Jsonb(event.drifted_features),
                event.reference_rows,
                event.current_rows,
                event.observed_at,
            ),
        ).fetchone()
    connection.commit()
    return row[0]


def maybe_create_retraining_request(
    connection,
    event: DriftEvent,
    report_id: int,
    *,
    consecutive_critical: int,
    cooldown_hours: int,
) -> int | None:
    with connection.cursor() as cursor:
        severity_rows = cursor.execute(
            """
            SELECT severity FROM drift_reports
            WHERE dataset_id = %s
            ORDER BY observed_at DESC
            LIMIT %s
            """,
            (event.dataset_id, consecutive_critical),
        ).fetchall()
        request_row = cursor.execute(
            """
            SELECT requested_at FROM retraining_requests
            WHERE dataset_id = %s
            ORDER BY requested_at DESC
            LIMIT 1
            """,
            (event.dataset_id,),
        ).fetchone()

        severities = [row[0] for row in severity_rows]
        last_request_at = request_row[0] if request_row else None
        if not should_request_retraining(
            severities,
            last_request_at,
            consecutive_critical=consecutive_critical,
            cooldown=timedelta(hours=cooldown_hours),
        ):
            return None

        existing = cursor.execute(
            """
            SELECT id FROM retraining_requests
            WHERE dataset_id = %s
              AND status IN ('pending_review', 'approved', 'running')
            ORDER BY requested_at DESC
            LIMIT 1
            """,
            (event.dataset_id,),
        ).fetchone()
        if existing:
            return None

        row = cursor.execute(
            """
            INSERT INTO retraining_requests (
                dataset_id, trigger, drift_report_id, notes
            ) VALUES (%s, 'persistent_critical_data_drift', %s, %s)
            RETURNING id
            """,
            (
                event.dataset_id,
                report_id,
                f"Created after {consecutive_critical} consecutive critical checks",
            ),
        ).fetchone()
    connection.commit()
    return row[0]


def run() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    bootstrap_servers = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
    topic = os.getenv("KAFKA_DRIFT_TOPIC", DRIFT_TOPIC)
    consecutive_critical = int(os.getenv("RETRAIN_CONSECUTIVE_CRITICAL", "3"))
    cooldown_hours = int(os.getenv("RETRAIN_COOLDOWN_HOURS", "24"))

    connection = connect_with_retry()
    ensure_schema(connection)
    consumer = KafkaConsumer(
        topic,
        bootstrap_servers=bootstrap_servers,
        group_id="retrain-coordinator",
        auto_offset_reset="earliest",
        enable_auto_commit=False,
        value_deserializer=lambda value: json.loads(value.decode()),
    )

    try:
        for message in consumer:
            try:
                event = DriftEvent.model_validate(message.value)
                report_id = save_drift_event(connection, event)
                request_id = maybe_create_retraining_request(
                    connection,
                    event,
                    report_id,
                    consecutive_critical=consecutive_critical,
                    cooldown_hours=cooldown_hours,
                )
                if request_id:
                    logger.warning(
                        "created retraining request=%s dataset=%s",
                        request_id,
                        event.dataset_id,
                    )
                consumer.commit()
            except (ValidationError, ValueError):
                logger.exception("Skipping invalid drift event")
                consumer.commit()
    finally:
        consumer.close()
        connection.close()


if __name__ == "__main__":
    run()

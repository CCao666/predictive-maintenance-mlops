"""Persist prediction events from Kafka to PostgreSQL."""

import json
import logging
import os

from kafka import KafkaConsumer

from streaming.consumer import RESULT_TOPIC
from streaming.database import connect_with_retry
from streaming.schemas import PredictionEvent


logger = logging.getLogger(__name__)


INSERT_PREDICTION = """
INSERT INTO rul_predictions (
    engine_id, time_cycle, predicted_rul, health_status, alert_level,
    model_name, model_version, model_alias, predicted_at
) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
ON CONFLICT (engine_id, time_cycle, model_version) DO UPDATE SET
    predicted_rul = EXCLUDED.predicted_rul,
    health_status = EXCLUDED.health_status,
    alert_level = EXCLUDED.alert_level,
    model_alias = EXCLUDED.model_alias,
    predicted_at = EXCLUDED.predicted_at
"""


def save_prediction(connection, event: PredictionEvent) -> None:
    with connection.cursor() as cursor:
        cursor.execute(
            INSERT_PREDICTION,
            (
                event.engine_id,
                event.time_cycle,
                event.predicted_rul,
                event.health_status,
                event.alert_level,
                event.model_name,
                event.model_version,
                event.model_alias,
                event.predicted_at,
            ),
        )
    connection.commit()


def run() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    bootstrap_servers = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
    topic = os.getenv("KAFKA_RESULT_TOPIC", RESULT_TOPIC)
    connection = connect_with_retry()
    consumer = KafkaConsumer(
        topic,
        bootstrap_servers=bootstrap_servers,
        group_id="prediction-writer",
        auto_offset_reset="earliest",
        enable_auto_commit=False,
        value_deserializer=lambda value: json.loads(value.decode()),
    )

    try:
        for message in consumer:
            event = PredictionEvent.model_validate(message.value)
            save_prediction(connection, event)
            consumer.commit()
            logger.info(
                "saved engine=%s cycle=%s", event.engine_id, event.time_cycle
            )
    finally:
        consumer.close()
        connection.close()


if __name__ == "__main__":
    run()

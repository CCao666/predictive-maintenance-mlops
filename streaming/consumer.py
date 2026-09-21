"""Consume sensor events, build windows, and request RUL predictions."""

import json
import logging
import os
import time

import httpx
from kafka import KafkaConsumer, KafkaProducer
from pydantic import ValidationError

from streaming.producer import DEFAULT_TOPIC
from streaming.schemas import PredictionEvent, SensorEvent, classify_rul
from streaming.window import EngineWindowStore


logger = logging.getLogger(__name__)
RESULT_TOPIC = "engine-rul-predictions"


def request_prediction(
    client: httpx.Client,
    api_url: str,
    engine_id: int,
    sequence: list[dict],
    attempts: int = 3,
) -> dict:
    """Call the inference API with bounded retries."""
    for attempt in range(1, attempts + 1):
        try:
            response = client.post(
                f"{api_url.rstrip('/')}/predict",
                json={"engine_id": engine_id, "sequence": sequence},
            )
            response.raise_for_status()
            return response.json()
        except httpx.HTTPError:
            if attempt == attempts:
                raise
            time.sleep(attempt)
    raise RuntimeError("Prediction retry loop ended unexpectedly")


def build_prediction_event(
    sensor_event: SensorEvent,
    result: dict,
) -> PredictionEvent:
    health_status, alert_level = classify_rul(result["predicted_rul"])
    return PredictionEvent(
        engine_id=sensor_event.engine_id,
        time_cycle=sensor_event.time_cycle,
        predicted_rul=result["predicted_rul"],
        health_status=health_status,
        alert_level=alert_level,
        model_name=result["model_name"],
        model_version=result["model_version"],
        model_alias=result["model_alias"],
    )


def run() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    bootstrap_servers = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
    topic = os.getenv("KAFKA_TOPIC", DEFAULT_TOPIC)
    result_topic = os.getenv("KAFKA_RESULT_TOPIC", RESULT_TOPIC)
    api_url = os.getenv("INFERENCE_API_URL", "http://localhost:8000")
    sequence_length = int(os.getenv("SEQUENCE_LENGTH", "50"))

    consumer = KafkaConsumer(
        topic,
        bootstrap_servers=bootstrap_servers,
        group_id="rul-inference",
        auto_offset_reset="earliest",
        enable_auto_commit=False,
        value_deserializer=lambda value: json.loads(value.decode()),
    )
    producer = KafkaProducer(
        bootstrap_servers=bootstrap_servers,
        key_serializer=lambda key: str(key).encode(),
        value_serializer=lambda value: json.dumps(value, default=str).encode(),
        acks="all",
    )
    windows = EngineWindowStore(sequence_length)

    try:
        with httpx.Client(timeout=10.0) as client:
            for message in consumer:
                try:
                    event = SensorEvent.model_validate(message.value)
                    sequence = windows.add(event)
                    if sequence is not None:
                        result = request_prediction(
                            client,
                            api_url,
                            event.engine_id,
                            sequence,
                        )
                        prediction = build_prediction_event(event, result)
                        producer.send(
                            result_topic,
                            key=event.engine_id,
                            value=prediction.model_dump(),
                        ).get(timeout=10)
                        logger.info(
                            "engine=%s cycle=%s predicted_rul=%.2f alert=%s",
                            event.engine_id,
                            event.time_cycle,
                            prediction.predicted_rul,
                            prediction.alert_level,
                        )
                except (ValidationError, ValueError):
                    logger.exception("Skipping invalid sensor event")
                except httpx.HTTPError:
                    logger.exception(
                        "Inference API unavailable; message not committed"
                    )
                    raise
                consumer.commit()
    finally:
        producer.close()
        consumer.close()


if __name__ == "__main__":
    run()

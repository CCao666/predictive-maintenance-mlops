"""Replay C-MAPSS rows into Kafka as sensor events."""

import argparse
import json
import os
import time
from collections.abc import Iterator
from pathlib import Path

from kafka import KafkaProducer

from streaming.schemas import SensorEvent
from training.dataset import FEATURE_COLUMNS, load_cmapss


DEFAULT_TOPIC = "engine-sensor-readings"


def iter_events(path: str | Path) -> Iterator[SensorEvent]:
    dataframe = load_cmapss(path).sort_values(["time_cycle", "unit_id"])
    for row in dataframe.itertuples(index=False):
        features = {name: getattr(row, name) for name in FEATURE_COLUMNS}
        yield SensorEvent(
            engine_id=row.unit_id,
            time_cycle=row.time_cycle,
            features=features,
        )


def replay(
    path: str | Path,
    bootstrap_servers: str,
    topic: str,
    delay: float = 0.0,
    limit: int | None = None,
) -> int:
    producer = KafkaProducer(
        bootstrap_servers=bootstrap_servers,
        key_serializer=lambda key: str(key).encode(),
        value_serializer=lambda value: json.dumps(value).encode(),
        acks="all",
    )

    sent = 0
    try:
        for event in iter_events(path):
            producer.send(
                topic,
                key=event.engine_id,
                value=event.model_dump(),
            )
            sent += 1
            if delay:
                time.sleep(delay)
            if limit is not None and sent >= limit:
                break
        producer.flush()
    finally:
        producer.close()
    return sent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        default="data/cmapss/test_FD001.txt",
        help="C-MAPSS file to replay",
    )
    parser.add_argument(
        "--bootstrap-servers",
        default=os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092"),
    )
    parser.add_argument(
        "--topic",
        default=os.getenv("KAFKA_TOPIC", DEFAULT_TOPIC),
    )
    parser.add_argument("--delay", type=float, default=0.0)
    parser.add_argument("--limit", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    count = replay(
        args.data,
        args.bootstrap_servers,
        args.topic,
        args.delay,
        args.limit,
    )
    print(f"Published {count} readings to {args.topic}")


if __name__ == "__main__":
    main()

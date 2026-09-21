"""Small PostgreSQL connection helper for streaming services."""

import os
import time

import psycopg


def database_url() -> str:
    return os.getenv(
        "DATABASE_URL",
        "postgresql://maintenance:maintenance@localhost:5432/maintenance",
    )


def connect_with_retry(attempts: int = 20, delay: float = 2.0):
    for attempt in range(1, attempts + 1):
        try:
            return psycopg.connect(database_url())
        except psycopg.OperationalError:
            if attempt == attempts:
                raise
            time.sleep(delay)
    raise RuntimeError("Database retry loop ended unexpectedly")

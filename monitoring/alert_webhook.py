"""Receive Alertmanager webhooks and persist alert state."""

from fastapi import FastAPI
from psycopg.types.json import Jsonb

from streaming.database import connect_with_retry


app = FastAPI(title="Local Alert Webhook")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "healthy"}


@app.post("/alerts")
def receive_alerts(payload: dict) -> dict[str, int]:
    alerts = payload.get("alerts", [])
    connection = connect_with_retry()
    try:
        with connection.cursor() as cursor:
            for alert in alerts:
                cursor.execute(
                    """
                    INSERT INTO alerts (
                        fingerprint, status, alert_name, severity, engine_id,
                        summary, description, labels, starts_at, ends_at,
                        last_received_at
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, NOW())
                    ON CONFLICT (fingerprint) DO UPDATE SET
                        status = EXCLUDED.status,
                        severity = EXCLUDED.severity,
                        summary = EXCLUDED.summary,
                        description = EXCLUDED.description,
                        labels = EXCLUDED.labels,
                        ends_at = EXCLUDED.ends_at,
                        last_received_at = NOW()
                    """,
                    (
                        alert["fingerprint"],
                        alert["status"],
                        alert["labels"].get("alertname", "unknown"),
                        alert["labels"].get("severity", "unknown"),
                        alert["labels"].get("engine_id"),
                        alert.get("annotations", {}).get("summary"),
                        alert.get("annotations", {}).get("description"),
                        Jsonb(alert["labels"]),
                        alert.get("startsAt"),
                        alert.get("endsAt"),
                    ),
                )
        connection.commit()
    finally:
        connection.close()
    return {"stored": len(alerts)}

CREATE TABLE IF NOT EXISTS rul_predictions (
    id BIGSERIAL PRIMARY KEY,
    engine_id INTEGER NOT NULL,
    time_cycle INTEGER NOT NULL,
    predicted_rul DOUBLE PRECISION NOT NULL,
    health_status TEXT NOT NULL,
    alert_level TEXT NOT NULL,
    model_name TEXT NOT NULL,
    model_version TEXT NOT NULL,
    model_alias TEXT NOT NULL,
    predicted_at TIMESTAMPTZ NOT NULL,
    UNIQUE (engine_id, time_cycle, model_version)
);

CREATE INDEX IF NOT EXISTS rul_predictions_engine_time_idx
    ON rul_predictions (engine_id, time_cycle);

CREATE TABLE IF NOT EXISTS alerts (
    fingerprint TEXT PRIMARY KEY,
    status TEXT NOT NULL,
    alert_name TEXT NOT NULL,
    severity TEXT NOT NULL,
    engine_id TEXT,
    summary TEXT,
    description TEXT,
    labels JSONB NOT NULL,
    starts_at TIMESTAMPTZ,
    ends_at TIMESTAMPTZ,
    last_received_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS alerts_status_severity_idx
    ON alerts (status, severity);

CREATE TABLE IF NOT EXISTS rul_predictions (
    id BIGSERIAL PRIMARY KEY,
    dataset_id TEXT NOT NULL DEFAULT 'FD001',
    engine_id INTEGER NOT NULL,
    time_cycle INTEGER NOT NULL,
    predicted_rul DOUBLE PRECISION NOT NULL,
    health_status TEXT NOT NULL,
    alert_level TEXT NOT NULL,
    model_name TEXT NOT NULL,
    model_version TEXT NOT NULL,
    model_alias TEXT NOT NULL,
    predicted_at TIMESTAMPTZ NOT NULL,
    UNIQUE (dataset_id, engine_id, time_cycle, model_version)
);

CREATE INDEX IF NOT EXISTS rul_predictions_engine_time_idx
    ON rul_predictions (dataset_id, engine_id, time_cycle);

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
);

CREATE INDEX IF NOT EXISTS drift_reports_dataset_time_idx
    ON drift_reports (dataset_id, observed_at DESC);

CREATE TABLE IF NOT EXISTS retraining_requests (
    id BIGSERIAL PRIMARY KEY,
    dataset_id TEXT NOT NULL,
    trigger TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending_review',
    drift_report_id BIGINT REFERENCES drift_reports(id),
    requested_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    notes TEXT,
    candidate_run_id TEXT,
    candidate_version TEXT,
    comparison JSONB,
    completed_at TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS retraining_requests_dataset_time_idx
    ON retraining_requests (dataset_id, requested_at DESC);

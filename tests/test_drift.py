"""Tests for offline C-MAPSS data-drift detection."""

from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from monitoring.drift import detect_data_drift
from monitoring.drift_monitor import RollingDriftMonitor, build_drift_event
from streaming.schemas import SensorEvent
from training.dataset import FEATURE_COLUMNS
from training.retrain_coordinator import should_request_retraining
from training.retrain_candidate import candidate_decision


def feature_frame(rows: int = 200, shift: float = 0.0) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    return pd.DataFrame({
        column: rng.normal(index, 1, rows) + shift
        for index, column in enumerate(FEATURE_COLUMNS)
    })


def test_identical_distributions_have_no_drift():
    reference = feature_frame()

    report = detect_data_drift(reference, reference.copy())

    assert report["severity"] == "none"
    assert report["drift_detected"] is False
    assert report["drifted_feature_count"] == 0
    assert report["overall_drift_score"] == 0


def test_shifted_distributions_are_critical():
    reference = feature_frame()
    current = feature_frame(shift=2.0)

    report = detect_data_drift(reference, current)

    assert report["severity"] == "critical"
    assert report["drift_detected"] is True
    assert report["drifted_feature_count"] == len(FEATURE_COLUMNS)
    assert report["overall_drift_score"] >= 0.20


def test_small_effect_is_not_reported_as_drift():
    reference = feature_frame(rows=5_000)
    current = feature_frame(rows=5_000, shift=0.05)

    report = detect_data_drift(reference, current)

    assert report["severity"] == "none"
    assert report["drifted_feature_count"] == 0


def test_missing_feature_is_rejected():
    reference = feature_frame()
    current = feature_frame().drop(columns=["sensor_21"])

    with pytest.raises(ValueError, match="Current data is missing columns"):
        detect_data_drift(reference, current)


def test_too_few_samples_is_rejected():
    reference = feature_frame(rows=20)
    current = feature_frame(rows=20)

    with pytest.raises(ValueError, match="at least 30"):
        detect_data_drift(reference, current)


def test_rolling_monitor_checks_only_after_interval():
    reference = feature_frame(rows=200)
    current = feature_frame(rows=30, shift=2.0)
    monitor = RollingDriftMonitor(
        reference,
        window_size=30,
        check_interval=30,
        min_samples=30,
    )

    for cycle, (_, row) in enumerate(current.iterrows(), start=1):
        report = monitor.add(SensorEvent(
            dataset_id="FD002",
            engine_id=1,
            time_cycle=cycle,
            features=row.to_dict(),
        ))

    assert report is not None
    event = build_drift_event("FD002", report)
    assert event.severity == "critical"
    assert event.current_rows == 30


def test_retraining_requires_persistent_critical_drift():
    assert not should_request_retraining(["critical", "critical"], None)
    assert not should_request_retraining(
        ["critical", "warning", "critical"],
        None,
    )
    assert should_request_retraining(
        ["critical", "critical", "critical"],
        None,
    )


def test_retraining_respects_cooldown():
    now = datetime(2026, 1, 2, tzinfo=timezone.utc)
    severities = ["critical", "critical", "critical"]

    assert not should_request_retraining(
        severities,
        now - timedelta(hours=23),
        now=now,
    )
    assert should_request_retraining(
        severities,
        now - timedelta(hours=24),
        now=now,
    )


def test_candidate_must_pass_every_promotion_gate():
    champion = {
        "rmse": 20.0,
        "nasa_score": 100.0,
        "severe_overestimate_count": 5,
    }
    eligible = candidate_decision(
        {
            "rmse": 18.0,
            "nasa_score": 90.0,
            "severe_overestimate_count": 4,
        },
        champion,
        {
            "rmse": 20.4,
            "nasa_score": 95.0,
            "severe_overestimate_count": 4,
        },
        champion,
    )
    unsafe = candidate_decision(
        {
            "rmse": 18.0,
            "nasa_score": 120.0,
            "severe_overestimate_count": 4,
        },
        champion,
        {
            "rmse": 20.4,
            "nasa_score": 95.0,
            "severe_overestimate_count": 4,
        },
        champion,
    )

    assert eligible["eligible"] is True
    assert unsafe["eligible"] is False


def test_candidate_is_rejected_when_fd001_regresses():
    champion = {
        "rmse": 20.0,
        "nasa_score": 100.0,
        "severe_overestimate_count": 5,
    }
    improved_on_drift = {
        "rmse": 18.0,
        "nasa_score": 90.0,
        "severe_overestimate_count": 4,
    }

    decision = candidate_decision(
        improved_on_drift,
        champion,
        {
            "rmse": 20.8,
            "nasa_score": 95.0,
            "severe_overestimate_count": 4,
        },
        champion,
    )

    assert decision["fd001_rmse_degradation"] == pytest.approx(0.04)
    assert decision["fd001_rmse_within_limit"] is False
    assert decision["eligible"] is False


@pytest.mark.parametrize(
    "fd001_candidate",
    [
        {"rmse": 20.0, "nasa_score": 101.0, "severe_overestimate_count": 5},
        {"rmse": 20.0, "nasa_score": 100.0, "severe_overestimate_count": 6},
    ],
)
def test_fd001_safety_metrics_cannot_worsen(fd001_candidate):
    champion = {
        "rmse": 20.0,
        "nasa_score": 100.0,
        "severe_overestimate_count": 5,
    }
    improved_on_drift = {
        "rmse": 18.0,
        "nasa_score": 90.0,
        "severe_overestimate_count": 4,
    }

    decision = candidate_decision(
        improved_on_drift,
        champion,
        fd001_candidate,
        champion,
    )

    assert decision["eligible"] is False

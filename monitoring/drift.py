"""Statistical data-drift detection for C-MAPSS features."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from training.dataset import FEATURE_COLUMNS


def detect_data_drift(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    *,
    p_value_threshold: float = 0.05,
    statistic_threshold: float = 0.20,
    warning_ratio: float = 0.20,
    critical_ratio: float = 0.75,
    min_samples: int = 30,
) -> dict[str, Any]:
    """Compare feature distributions with two-sample KS tests.

    A feature is drifted only when the difference is statistically significant
    and its KS statistic is large enough to matter in practice.
    """
    _validate_inputs(
        reference,
        current,
        p_value_threshold,
        statistic_threshold,
        warning_ratio,
        critical_ratio,
        min_samples,
    )

    feature_results: dict[str, dict[str, float | int | bool]] = {}

    for column in FEATURE_COLUMNS:
        reference_values = reference[column].dropna().to_numpy()
        current_values = current[column].dropna().to_numpy()

        if min(len(reference_values), len(current_values)) < min_samples:
            raise ValueError(
                f"Feature {column!r} needs at least {min_samples} "
                "non-null samples in each dataset"
            )

        statistic, p_value = ks_2samp(reference_values, current_values)
        drifted = (
            p_value < p_value_threshold
            and statistic >= statistic_threshold
        )
        feature_results[column] = {
            "ks_statistic": float(statistic),
            "p_value": float(p_value),
            "drifted": bool(drifted),
            "reference_samples": len(reference_values),
            "current_samples": len(current_values),
        }

    statistics = np.array([
        result["ks_statistic"] for result in feature_results.values()
    ])
    drifted_features = [
        column
        for column, result in feature_results.items()
        if result["drifted"]
    ]
    drifted_ratio = len(drifted_features) / len(FEATURE_COLUMNS)

    if drifted_ratio >= critical_ratio:
        severity = "critical"
    elif drifted_ratio >= warning_ratio:
        severity = "warning"
    else:
        severity = "none"

    return {
        "drift_detected": severity != "none",
        "severity": severity,
        "reference_rows": len(reference),
        "current_rows": len(current),
        "feature_count": len(FEATURE_COLUMNS),
        "drifted_feature_count": len(drifted_features),
        "drifted_feature_ratio": float(drifted_ratio),
        "drifted_features": drifted_features,
        "overall_drift_score": float(statistics.max()),
        "mean_ks_statistic": float(statistics.mean()),
        "median_ks_statistic": float(np.median(statistics)),
        "thresholds": {
            "p_value": p_value_threshold,
            "ks_statistic": statistic_threshold,
            "warning_ratio": warning_ratio,
            "critical_ratio": critical_ratio,
            "min_samples": min_samples,
        },
        "features": feature_results,
    }


def _validate_inputs(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    p_value_threshold: float,
    statistic_threshold: float,
    warning_ratio: float,
    critical_ratio: float,
    min_samples: int,
) -> None:
    missing_reference = set(FEATURE_COLUMNS) - set(reference.columns)
    missing_current = set(FEATURE_COLUMNS) - set(current.columns)

    if missing_reference:
        raise ValueError(
            f"Reference data is missing columns: {sorted(missing_reference)}"
        )
    if missing_current:
        raise ValueError(
            f"Current data is missing columns: {sorted(missing_current)}"
        )
    if reference.empty or current.empty:
        raise ValueError("Reference and current data must not be empty")
    if not 0 < p_value_threshold < 1:
        raise ValueError("p_value_threshold must be between 0 and 1")
    if not 0 <= statistic_threshold <= 1:
        raise ValueError("statistic_threshold must be between 0 and 1")
    if not 0 <= warning_ratio < critical_ratio <= 1:
        raise ValueError(
            "severity ratios must satisfy 0 <= warning < critical <= 1"
        )
    if min_samples < 2:
        raise ValueError("min_samples must be at least 2")

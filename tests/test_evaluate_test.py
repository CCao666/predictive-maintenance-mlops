import numpy as np
import pandas as pd

from training.dataset import FEATURE_COLUMNS
from training.evaluate_test import (
    build_test_sequences,
    classification_metrics,
    nasa_scores,
    regression_metrics,
)


class IdentityScaler:
    def transform(self, values):
        return np.asarray(values)


def test_build_test_sequences_left_pads_short_engines():
    rows = []
    for cycle in (1, 2):
        rows.append({
            "unit_id": 1,
            "time_cycle": cycle,
            **{name: float(cycle) for name in FEATURE_COLUMNS},
        })

    engine_ids, sequences = build_test_sequences(
        pd.DataFrame(rows),
        IdentityScaler(),
        sequence_length=3,
    )

    assert engine_ids == [1]
    assert sequences.shape == (1, 3, 24)
    assert np.all(sequences[0, 0] == 0)
    assert np.all(sequences[0, -1] == 2)


def test_nasa_score_penalizes_dangerous_overestimate_more():
    targets = np.array([50.0])
    overestimate = nasa_scores(targets, np.array([60.0]))[0]
    underestimate = nasa_scores(targets, np.array([40.0]))[0]

    assert overestimate > underestimate


def test_evaluation_metrics_capture_business_risk():
    targets = np.array([10.0, 20.0, 80.0])
    predictions = np.array([15.0, 40.0, 70.0])

    regression = regression_metrics(targets, predictions)
    business = classification_metrics(targets, predictions)

    assert regression["severe_overestimate_count"] == 0
    assert business["critical_recall"] == 0.5
    assert business["missed_critical_engines"] == 1

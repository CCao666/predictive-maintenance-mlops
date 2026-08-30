"""Feature preprocessing for LSTM training."""

from __future__ import annotations

import pandas as pd
from sklearn.preprocessing import RobustScaler

from training.dataset import FEATURE_COLUMNS


def fit_feature_scaler(
    train_dataframe: pd.DataFrame,
) -> RobustScaler:
    """Fit a RobustScaler using training features only."""
    missing_columns = set(FEATURE_COLUMNS).difference(
        train_dataframe.columns
    )

    if missing_columns:
        raise ValueError(
            f"DataFrame is missing columns: {sorted(missing_columns)}"
        )

    if train_dataframe.empty:
        raise ValueError("train_dataframe must not be empty")

    scaler = RobustScaler()
    scaler.fit(train_dataframe[FEATURE_COLUMNS])

    return scaler


def transform_features(
    dataframe: pd.DataFrame,
    scaler: RobustScaler,
) -> pd.DataFrame:
    """Return a copy with robust-scaled model features."""
    missing_columns = set(FEATURE_COLUMNS).difference(
        dataframe.columns
    )

    if missing_columns:
        raise ValueError(
            f"DataFrame is missing columns: {sorted(missing_columns)}"
        )

    result = dataframe.copy()

    transformed_features = scaler.transform(
        result[FEATURE_COLUMNS]
    )

    result[FEATURE_COLUMNS] = pd.DataFrame(
        transformed_features,
        columns=FEATURE_COLUMNS,
        index=result.index,
    )

    return result

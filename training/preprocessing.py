"""Feature preprocessing for LSTM training."""

import pandas as pd
from sklearn.preprocessing import RobustScaler

from training.dataset import FEATURE_COLUMNS


def fit_feature_scaler(
    train_dataframe: pd.DataFrame,
) -> RobustScaler:
    """Fit a RobustScaler using training features only."""
    if train_dataframe.empty:
        raise ValueError("train_dataframe must not be empty")
    _check_features(train_dataframe)

    return RobustScaler().fit(train_dataframe[FEATURE_COLUMNS])


def transform_features(
    dataframe: pd.DataFrame,
    scaler: RobustScaler,
) -> pd.DataFrame:
    """Return a copy with robust-scaled model features."""
    _check_features(dataframe)
    result = dataframe.copy()
    result[FEATURE_COLUMNS] = pd.DataFrame(
        scaler.transform(result[FEATURE_COLUMNS]),
        columns=FEATURE_COLUMNS,
        index=result.index,
    )
    return result


def _check_features(dataframe: pd.DataFrame) -> None:
    missing = set(FEATURE_COLUMNS) - set(dataframe.columns)
    if missing:
        raise ValueError(f"DataFrame is missing columns: {sorted(missing)}")

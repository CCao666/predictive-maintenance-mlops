"""Engine-level train and validation split."""

import pandas as pd
from sklearn.model_selection import train_test_split


def split_by_engine(
    dataframe: pd.DataFrame,
    validation_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Keep every engine entirely in either train or validation data."""
    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError("dataframe must be a pandas DataFrame")

    if dataframe.empty:
        raise ValueError("dataframe must not be empty")

    if "unit_id" not in dataframe.columns:
        raise ValueError("dataframe must contain a unit_id column")

    if dataframe["unit_id"].isna().any():
        raise ValueError("unit_id must not contain missing values")

    if not 0 < validation_size < 1:
        raise ValueError("validation_size must be between 0 and 1")

    engine_ids = sorted(dataframe["unit_id"].unique())

    if len(engine_ids) < 2:
        raise ValueError(
            "At least two unique engines are required for splitting"
        )

    train_engine_ids, validation_engine_ids = train_test_split(
        engine_ids,
        test_size=validation_size,
        random_state=random_state,
        shuffle=True,
    )

    train = dataframe[dataframe["unit_id"].isin(train_engine_ids)]
    validation = dataframe[dataframe["unit_id"].isin(validation_engine_ids)]
    return train.reset_index(drop=True), validation.reset_index(drop=True)

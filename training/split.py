"""Engine-level train and validation splitting utilities."""

from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split


def split_by_engine(
    dataframe: pd.DataFrame,
    validation_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split a C-MAPSS DataFrame by engine ID.

    All rows belonging to one engine are placed entirely in either the
    training set or validation set. This prevents data leakage between
    the two sets.

    Args:
        dataframe:
            C-MAPSS DataFrame containing a ``unit_id`` column.
        validation_size:
            Fraction of engines assigned to validation. Must be between
            0 and 1.
        random_state:
            Random seed used to make the split reproducible.

    Returns:
        A tuple containing ``(train_dataframe, validation_dataframe)``.

    Raises:
        TypeError:
            If dataframe is not a pandas DataFrame.
        ValueError:
            If required data is missing or validation_size is invalid.
    """
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

    train_engine_ids = set(train_engine_ids)
    validation_engine_ids = set(validation_engine_ids)

    train_dataframe = dataframe[
        dataframe["unit_id"].isin(train_engine_ids)
    ].copy()

    validation_dataframe = dataframe[
        dataframe["unit_id"].isin(validation_engine_ids)
    ].copy()

    # Reset indexes so both returned DataFrames have clean, independent indexes.
    train_dataframe = train_dataframe.reset_index(drop=True)
    validation_dataframe = validation_dataframe.reset_index(drop=True)

    return train_dataframe, validation_dataframe

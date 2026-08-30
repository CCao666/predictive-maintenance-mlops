"""Tests for LSTM feature preprocessing."""

import numpy as np
import pandas as pd
import pytest

from training.dataset import (
    FEATURE_COLUMNS,
    add_train_rul,
    load_cmapss,
)
from training.preprocessing import (
    fit_feature_scaler,
    transform_features,
)
from training.split import split_by_engine


DATA_PATH = "data/cmapss/train_FD001.txt"


@pytest.fixture
def split_dataframes():
    dataframe = add_train_rul(load_cmapss(DATA_PATH))

    return split_by_engine(
        dataframe,
        validation_size=0.2,
        random_state=42,
    )


def test_scaled_training_features_have_zero_median(
    split_dataframes,
):
    train_df, _ = split_dataframes

    scaler = fit_feature_scaler(train_df)
    scaled_train_df = transform_features(
        train_df,
        scaler,
    )

    feature_medians = scaled_train_df[
        FEATURE_COLUMNS
    ].median()

    assert np.allclose(
        feature_medians,
        0.0,
        atol=1e-6,
    )


def test_transform_does_not_modify_original(
    split_dataframes,
):
    train_df, _ = split_dataframes
    original = train_df.copy(deep=True)

    scaler = fit_feature_scaler(train_df)
    transform_features(train_df, scaler)

    pd.testing.assert_frame_equal(train_df, original)


def test_non_feature_columns_are_preserved(
    split_dataframes,
):
    train_df, _ = split_dataframes

    scaler = fit_feature_scaler(train_df)
    scaled_train_df = transform_features(
        train_df,
        scaler,
    )

    assert scaled_train_df["unit_id"].equals(
        train_df["unit_id"]
    )
    assert scaled_train_df["time_cycle"].equals(
        train_df["time_cycle"]
    )
    assert scaled_train_df["rul"].equals(
        train_df["rul"]
    )


def test_same_scaler_transforms_validation_data(
    split_dataframes,
):
    train_df, validation_df = split_dataframes

    scaler = fit_feature_scaler(train_df)

    scaled_validation_df = transform_features(
        validation_df,
        scaler,
    )

    assert len(scaled_validation_df) == len(
        validation_df
    )
    assert not scaled_validation_df[
        FEATURE_COLUMNS
    ].isna().any().any()


def test_scaler_rejects_missing_features():
    dataframe = pd.DataFrame(
        {
            "unit_id": [1],
            "time_cycle": [1],
        }
    )

    with pytest.raises(ValueError):
        fit_feature_scaler(dataframe)

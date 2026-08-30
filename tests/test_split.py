"""Tests for engine-level train and validation splitting."""

import pandas as pd
import pytest

from training.dataset import add_train_rul, load_cmapss
from training.split import split_by_engine


DATA_PATH = "data/cmapss/train_FD001.txt"


@pytest.fixture
def labeled_dataframe():
    dataframe = load_cmapss(DATA_PATH)
    return add_train_rul(dataframe)


def test_split_by_engine_counts(labeled_dataframe):
    train_df, validation_df = split_by_engine(
        labeled_dataframe,
        validation_size=0.2,
        random_state=42,
    )

    assert train_df["unit_id"].nunique() == 80
    assert validation_df["unit_id"].nunique() == 20


def test_split_has_no_engine_overlap(labeled_dataframe):
    train_df, validation_df = split_by_engine(
        labeled_dataframe,
        validation_size=0.2,
        random_state=42,
    )

    train_ids = set(train_df["unit_id"])
    validation_ids = set(validation_df["unit_id"])

    assert train_ids.isdisjoint(validation_ids)


def test_split_preserves_every_row(labeled_dataframe):
    train_df, validation_df = split_by_engine(
        labeled_dataframe,
        validation_size=0.2,
        random_state=42,
    )

    assert len(train_df) + len(validation_df) == len(
        labeled_dataframe
    )


def test_split_is_reproducible(labeled_dataframe):
    first_train, first_validation = split_by_engine(
        labeled_dataframe,
        random_state=42,
    )

    second_train, second_validation = split_by_engine(
        labeled_dataframe,
        random_state=42,
    )

    assert set(first_train["unit_id"]) == set(
        second_train["unit_id"]
    )
    assert set(first_validation["unit_id"]) == set(
        second_validation["unit_id"]
    )


def test_split_rejects_invalid_validation_size(
    labeled_dataframe,
):
    with pytest.raises(ValueError):
        split_by_engine(
            labeled_dataframe,
            validation_size=0,
        )

    with pytest.raises(ValueError):
        split_by_engine(
            labeled_dataframe,
            validation_size=1,
        )


def test_split_rejects_missing_unit_id():
    dataframe = pd.DataFrame({"sensor_1": [1.0, 2.0]})

    with pytest.raises(ValueError):
        split_by_engine(dataframe)
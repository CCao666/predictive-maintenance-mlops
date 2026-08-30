"""Tests for C-MAPSS loading, RUL labels, and sequence generation."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from training.dataset import (
    FEATURE_COLUMNS,
    add_train_rul,
    create_sequences,
    load_cmapss,
)


DATA_PATH = Path("data/cmapss/train_FD001.txt")


def test_load_cmapss():
    dataframe = load_cmapss(DATA_PATH)

    assert len(dataframe) == 20_631
    assert dataframe.shape[1] == 26
    assert dataframe["unit_id"].nunique() == 100
    assert dataframe.isna().sum().sum() == 0


def test_load_cmapss_rejects_missing_file():
    with pytest.raises(FileNotFoundError):
        load_cmapss("data/cmapss/does_not_exist.txt")


def test_add_train_rul():
    dataframe = load_cmapss(DATA_PATH)
    dataframe = add_train_rul(dataframe)

    assert "rul" in dataframe.columns
    assert dataframe["rul"].min() == 0

    final_rows = dataframe.loc[
        dataframe.groupby("unit_id")["time_cycle"].idxmax()
    ]

    assert (final_rows["rul"] == 0).all()


def test_add_train_rul_does_not_modify_original():
    original = pd.DataFrame(
        {
            "unit_id": [1, 1],
            "time_cycle": [1, 2],
        }
    )

    result = add_train_rul(original)

    assert "rul" not in original.columns
    assert result["rul"].tolist() == [1, 0]


def test_create_sequences():
    dataframe = add_train_rul(load_cmapss(DATA_PATH))

    sequences, targets = create_sequences(
        dataframe,
        sequence_length=30,
    )

    assert sequences.shape == (17_731, 30, 24)
    assert targets.shape == (17_731,)
    assert sequences.dtype == np.float32
    assert targets.dtype == np.float32


def test_sequences_do_not_cross_engines():
    dataframe = add_train_rul(load_cmapss(DATA_PATH))

    engine_1 = dataframe[dataframe["unit_id"] == 1]
    sequences, targets = create_sequences(
        engine_1,
        sequence_length=30,
    )

    expected_windows = len(engine_1) - 30 + 1

    assert len(sequences) == expected_windows
    assert len(targets) == expected_windows
    assert targets[0] == engine_1.iloc[29]["rul"]


def test_short_engine_returns_empty_arrays():
    columns = ["unit_id", "time_cycle", "rul", *FEATURE_COLUMNS]
    dataframe = pd.DataFrame(
        [[1, 1, 10, *([0.0] * 24)]],
        columns=columns,
    )

    sequences, targets = create_sequences(
        dataframe,
        sequence_length=30,
    )

    assert sequences.shape == (0, 30, 24)
    assert targets.shape == (0,)

"""Load C-MAPSS data, calculate RUL labels, and create sequences."""

from pathlib import Path

import numpy as np
import pandas as pd


COLUMN_NAMES = [
    "unit_id",
    "time_cycle",
    "op_setting_1",
    "op_setting_2",
    "op_setting_3",
    *[f"sensor_{i}" for i in range(1, 22)],
]

FEATURE_COLUMNS = [
    "op_setting_1",
    "op_setting_2",
    "op_setting_3",
    *[f"sensor_{i}" for i in range(1, 22)],
]


def load_cmapss(path: str | Path) -> pd.DataFrame:
    """Load a C-MAPSS sensor data file.

    Each row contains:
    unit_id, time_cycle, 3 operational settings, and 21 sensors.
    """
    path = Path(path)

    if not path.is_file():
        raise FileNotFoundError(f"C-MAPSS file not found: {path}")

    dataframe = pd.read_csv(
        path,
        sep=r"\s+",
        header=None,
        names=COLUMN_NAMES,
    )

    if dataframe.shape[1] != len(COLUMN_NAMES):
        raise ValueError(
            f"Expected {len(COLUMN_NAMES)} columns, "
            f"but found {dataframe.shape[1]}"
        )

    return dataframe


def add_train_rul(
    dataframe: pd.DataFrame,
    max_rul: int | None = None,
) -> pd.DataFrame:
    """Calculate RUL for complete run-to-failure training trajectories.

    RUL = final cycle for the engine - current cycle
    """
    required_columns = {"unit_id", "time_cycle"}

    if not required_columns.issubset(dataframe.columns):
        raise ValueError(
            "DataFrame must contain unit_id and time_cycle columns"
        )

    result = dataframe.copy()

    max_cycles = result.groupby("unit_id")["time_cycle"].transform("max")
    result["rul"] = max_cycles - result["time_cycle"]

    if max_rul is not None:
        if max_rul <= 0:
            raise ValueError("max_rul must be greater than zero")
        result["rul"] = result["rul"].clip(upper=max_rul)

    return result


def create_sequences(
    dataframe: pd.DataFrame,
    sequence_length: int = 30,
) -> tuple[np.ndarray, np.ndarray]:
    """Create fixed-length sequences without crossing engine boundaries.

    The target is the RUL value at the final cycle of each sequence.
    """
    if sequence_length <= 0:
        raise ValueError("sequence_length must be greater than zero")

    required_columns = {"unit_id", "time_cycle", "rul", *FEATURE_COLUMNS}
    missing_columns = required_columns.difference(dataframe.columns)

    if missing_columns:
        raise ValueError(
            f"DataFrame is missing columns: {sorted(missing_columns)}"
        )

    sequences = []
    targets = []

    for _, engine_data in dataframe.groupby("unit_id", sort=True):
        engine_data = engine_data.sort_values("time_cycle")

        feature_values = engine_data[FEATURE_COLUMNS].to_numpy(
            dtype=np.float32
        )
        rul_values = engine_data["rul"].to_numpy(dtype=np.float32)

        number_of_windows = len(engine_data) - sequence_length + 1

        for start_index in range(max(0, number_of_windows)):
            end_index = start_index + sequence_length

            sequences.append(feature_values[start_index:end_index])
            targets.append(rul_values[end_index - 1])

    if not sequences:
        return (
            np.empty(
                (0, sequence_length, len(FEATURE_COLUMNS)),
                dtype=np.float32,
            ),
            np.empty((0,), dtype=np.float32),
        )

    return (
        np.stack(sequences),
        np.asarray(targets, dtype=np.float32),
    )

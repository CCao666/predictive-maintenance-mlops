"""Inspect and validate the C-MAPSS dataset from the command line."""
from pathlib import Path

from training.dataset import (
    add_train_rul,
    create_sequences,
    load_cmapss,
)


data_path = Path("data/cmapss/train_FD001.txt")

dataframe = load_cmapss(data_path)
dataframe = add_train_rul(dataframe)
sequences, targets = create_sequences(dataframe, sequence_length=30)

print(dataframe.head())
print()
print(f"Rows: {len(dataframe)}")
print(f"Engines: {dataframe['unit_id'].nunique()}")
print(f"Sequence shape: {sequences.shape}")
print(f"Target shape: {targets.shape}")
print(f"First target: {targets[0]}")

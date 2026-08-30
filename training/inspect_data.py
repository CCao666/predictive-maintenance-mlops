from pathlib import Path

from training.dataset import add_train_rul, load_cmapss
from training.split import split_by_engine


dataframe = load_cmapss(
    Path("data/cmapss/train_FD001.txt")
)
dataframe = add_train_rul(dataframe)

train_dataframe, validation_dataframe = split_by_engine(
    dataframe,
    validation_size=0.2,
    random_state=42,
)

train_engines = set(train_dataframe["unit_id"])
validation_engines = set(validation_dataframe["unit_id"])

print(f"Training engines: {len(train_engines)}")
print(f"Validation engines: {len(validation_engines)}")
print(f"Training rows: {len(train_dataframe)}")
print(f"Validation rows: {len(validation_dataframe)}")
print(
    "Overlapping engines:",
    train_engines.intersection(validation_engines),
)
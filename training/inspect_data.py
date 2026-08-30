from training.dataset import (
    add_train_rul,
    create_sequences,
    load_cmapss,
)
from training.preprocessing import (
    fit_feature_scaler,
    transform_features,
)
from training.split import split_by_engine


dataframe = add_train_rul(
    load_cmapss("data/cmapss/train_FD001.txt"),
    max_rul=125,
)

train_df, validation_df = split_by_engine(
    dataframe,
    validation_size=0.2,
    random_state=42,
)

scaler = fit_feature_scaler(train_df)

scaled_train_df = transform_features(
    train_df,
    scaler,
)
scaled_validation_df = transform_features(
    validation_df,
    scaler,
)

X_train, y_train = create_sequences(
    scaled_train_df,
    sequence_length=50,
)

X_validation, y_validation = create_sequences(
    scaled_validation_df,
    sequence_length=50,
)

print(f"X_train: {X_train.shape}")
print(f"y_train: {y_train.shape}")
print(f"X_validation: {X_validation.shape}")
print(f"y_validation: {y_validation.shape}")

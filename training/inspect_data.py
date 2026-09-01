from training.train_lstm import prepare_datasets


def main() -> None:
    train_data, validation_data, scaler = prepare_datasets()
    print(f"Training sequences: {train_data.sequences.shape}")
    print(f"Training targets: {train_data.targets.shape}")
    print(f"Validation sequences: {validation_data.sequences.shape}")
    print(f"Validation targets: {validation_data.targets.shape}")
    print(f"Scaler: {type(scaler).__name__}")


if __name__ == "__main__":
    main()

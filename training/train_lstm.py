"""Train the upstream-equivalent PyTorch LSTM RUL reference model."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import joblib
import numpy as np
import torch
from sklearn.preprocessing import RobustScaler
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from training.dataset import (
    FEATURE_COLUMNS,
    add_train_rul,
    create_sequences,
    load_cmapss,
)
from training.model import ReferenceLSTMRegressor
from training.mlflow_tracking import (
    DEFAULT_EXPERIMENT_NAME,
    DEFAULT_TRACKING_URI,
    MLflowExperimentTracker,
)
from training.preprocessing import fit_feature_scaler, transform_features
from training.split import split_by_engine
from training.torch_dataset import RULSequenceDataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_PATH = PROJECT_ROOT / "data/cmapss/train_FD001.txt"
ARTIFACTS_DIRECTORY = PROJECT_ROOT / "artifacts"

DEFAULT_SEQUENCE_LENGTH = 50
DEFAULT_MAX_RUL = 125
DEFAULT_BATCH_SIZE = 32
DEFAULT_EPOCHS = 200
DEFAULT_LEARNING_RATE = 1e-3


def set_random_seed(random_state: int) -> None:
    """Seed Python, NumPy, and PyTorch for reproducible experiments."""
    random.seed(random_state)
    np.random.seed(random_state)
    torch.manual_seed(random_state)


def select_device() -> torch.device:
    """Use Apple Silicon, CUDA, or CPU in that order when available."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def prepare_datasets(
    data_path: Path = DEFAULT_DATA_PATH,
    sequence_length: int = DEFAULT_SEQUENCE_LENGTH,
    max_rul: int = DEFAULT_MAX_RUL,
    validation_size: float = 0.2,
    random_state: int = 42,
) -> tuple[RULSequenceDataset, RULSequenceDataset, RobustScaler]:
    """Load, split, robust-scale, and sequence the FD001 data."""
    dataframe = add_train_rul(
        load_cmapss(data_path),
        max_rul=max_rul,
    )
    train_df, validation_df = split_by_engine(
        dataframe,
        validation_size=validation_size,
        random_state=random_state,
    )

    scaler = fit_feature_scaler(train_df)
    scaled_train_df = transform_features(train_df, scaler)
    scaled_validation_df = transform_features(validation_df, scaler)

    X_train, y_train = create_sequences(
        scaled_train_df,
        sequence_length=sequence_length,
    )
    X_validation, y_validation = create_sequences(
        scaled_validation_df,
        sequence_length=sequence_length,
    )

    return (
        RULSequenceDataset(X_train, y_train),
        RULSequenceDataset(X_validation, y_validation),
        scaler,
    )


def train_one_epoch(
    model: nn.Module,
    data_loader: DataLoader,
    optimizer: Adam,
    loss_function: nn.Module,
    device: torch.device,
) -> float:
    """Train for one epoch and return the sample-weighted MSE."""
    model.train()
    total_loss = 0.0
    total_samples = 0

    for sequences, targets in data_loader:
        sequences = sequences.to(device)
        targets = targets.to(device)
        optimizer.zero_grad()
        predictions = model(sequences)
        loss = loss_function(predictions, targets)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        batch_size = len(targets)
        total_loss += loss.item() * batch_size
        total_samples += batch_size

    return total_loss / total_samples


def evaluate(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    """Return MAE and RMSE measured in RUL cycles."""
    model.eval()
    predictions = []
    targets = []

    with torch.no_grad():
        for sequences, batch_targets in data_loader:
            predictions.append(model(sequences.to(device)).cpu())
            targets.append(batch_targets)

    errors = torch.cat(predictions) - torch.cat(targets)
    return {
        "mae": torch.mean(torch.abs(errors)).item(),
        "rmse": torch.sqrt(torch.mean(errors**2)).item(),
    }


def train_lstm(
    data_path: Path = DEFAULT_DATA_PATH,
    sequence_length: int = DEFAULT_SEQUENCE_LENGTH,
    max_rul: int = DEFAULT_MAX_RUL,
    batch_size: int = DEFAULT_BATCH_SIZE,
    epochs: int = DEFAULT_EPOCHS,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    validation_size: float = 0.2,
    random_state: int = 42,
    early_stopping_patience: int = 20,
    lr_patience: int = 10,
    tracker: MLflowExperimentTracker | None = None,
) -> tuple[ReferenceLSTMRegressor, RobustScaler, dict[str, object]]:
    """Train the reference model and restore its best validation weights."""
    if epochs <= 0:
        raise ValueError("epochs must be greater than zero")

    set_random_seed(random_state)
    device = select_device()
    train_dataset, validation_dataset, scaler = prepare_datasets(
        data_path=data_path,
        sequence_length=sequence_length,
        max_rul=max_rul,
        validation_size=validation_size,
        random_state=random_state,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    model = ReferenceLSTMRegressor(input_size=len(FEATURE_COLUMNS)).to(device)
    optimizer = Adam(model.parameters(), lr=learning_rate)
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=lr_patience,
        min_lr=1e-5,
    )
    loss_function = nn.MSELoss()

    best_rmse = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    best_state = None
    history = []

    print(f"Device: {device}")
    print(f"Training sequences: {len(train_dataset)}")
    print(f"Validation sequences: {len(validation_dataset)}")

    for epoch in range(1, epochs + 1):
        train_mse = train_one_epoch(
            model,
            train_loader,
            optimizer,
            loss_function,
            device,
        )
        validation_metrics = evaluate(model, validation_loader, device)
        scheduler.step(validation_metrics["rmse"])
        current_lr = optimizer.param_groups[0]["lr"]

        epoch_metrics = {
            "epoch": epoch,
            "train_rmse": train_mse**0.5,
            "validation_mae": validation_metrics["mae"],
            "validation_rmse": validation_metrics["rmse"],
            "learning_rate": current_lr,
        }
        history.append(epoch_metrics)
        if tracker is not None:
            tracker.log_epoch(epoch_metrics, step=epoch)
        print(
            f"Epoch {epoch:03d}/{epochs} | "
            f"train RMSE {epoch_metrics['train_rmse']:.2f} | "
            f"validation MAE {epoch_metrics['validation_mae']:.2f} | "
            f"validation RMSE {epoch_metrics['validation_rmse']:.2f} | "
            f"lr {current_lr:.6f}"
        )

        if validation_metrics["rmse"] < best_rmse - 0.001:
            best_rmse = validation_metrics["rmse"]
            best_epoch = epoch
            epochs_without_improvement = 0
            best_state = {
                name: parameter.detach().cpu().clone()
                for name, parameter in model.state_dict().items()
            }
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= early_stopping_patience:
            print(f"Early stopping at epoch {epoch}; best epoch was {best_epoch}")
            break

    if best_state is None:
        raise RuntimeError("Training completed without producing model weights")

    model.load_state_dict(best_state)
    model = model.cpu()
    metadata = {
        "model_name": "reference_lstm_rul_predictor",
        "source_architecture": "upstream-reference TensorFlow model, ported to PyTorch",
        "model_config": model.get_config(),
        "preprocessing": {
            "scaler": "RobustScaler",
            "feature_columns": FEATURE_COLUMNS,
            "max_rul": max_rul,
            "sequence_length": sequence_length,
        },
        "training": {
            "batch_size": batch_size,
            "maximum_epochs": epochs,
            "completed_epochs": len(history),
            "learning_rate": learning_rate,
            "validation_size": validation_size,
            "random_state": random_state,
            "early_stopping_patience": early_stopping_patience,
            "lr_patience": lr_patience,
        },
        "best_epoch": best_epoch,
        "best_validation_rmse": best_rmse,
        "history": history,
    }
    return model, scaler, metadata


def save_artifacts(
    model: ReferenceLSTMRegressor,
    scaler: RobustScaler,
    metadata: dict[str, object],
    artifacts_directory: Path = ARTIFACTS_DIRECTORY,
) -> dict[str, Path]:
    """Save a self-describing model checkpoint, scaler, and metadata."""
    artifacts_directory.mkdir(parents=True, exist_ok=True)
    model_path = artifacts_directory / "reference_lstm_model.pt"
    scaler_path = artifacts_directory / "robust_scaler.joblib"
    metadata_path = artifacts_directory / "training_metrics.json"

    torch.save(
        {
            "model_class": "ReferenceLSTMRegressor",
            "model_config": model.get_config(),
            "state_dict": model.state_dict(),
        },
        model_path,
    )
    joblib.dump(scaler, scaler_path)
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"Saved model: {model_path}")
    print(f"Saved scaler: {scaler_path}")
    print(f"Saved metrics: {metadata_path}")
    return {
        "model": model_path,
        "scaler": scaler_path,
        "metadata": metadata_path,
    }


def parse_args() -> argparse.Namespace:
    """Parse command-line training options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--sequence-length", type=int, default=DEFAULT_SEQUENCE_LENGTH)
    parser.add_argument("--max-rul", type=int, default=DEFAULT_MAX_RUL)
    parser.add_argument("--experiment-name", default=DEFAULT_EXPERIMENT_NAME)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--tracking-uri", default=DEFAULT_TRACKING_URI)
    parser.add_argument("--disable-mlflow", action="store_true")
    return parser.parse_args()


def run_tracked_training(
    *,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    sequence_length: int,
    max_rul: int,
    tracking_uri: str,
    experiment_name: str,
    run_name: str | None,
    enable_mlflow: bool = True,
    tags: dict[str, str] | None = None,
) -> str | None:
    """Run one training experiment and return its MLflow run ID."""
    tracker = None
    run_id = None
    if enable_mlflow:
        tracker = MLflowExperimentTracker(
            tracking_uri=tracking_uri,
            experiment_name=experiment_name,
        )
        run_id = tracker.start_run(
            run_name=run_name,
            tags={
                "model_type": "ReferenceLSTMRegressor",
                "task": "remaining_useful_life_regression",
                "dataset": "NASA_CMAPSS_FD001",
                "framework": "pytorch",
                **(tags or {}),
            },
        )
        tracker.log_params(
            {
                "sequence_length": sequence_length,
                "max_rul": max_rul,
                "batch_size": batch_size,
                "maximum_epochs": epochs,
                "learning_rate": learning_rate,
                "random_state": 42,
                "validation_size": 0.2,
                "scaler": "RobustScaler",
                "lstm_hidden_sizes": "128,64,32",
                "attention_units": 64,
                "dense_sizes": "64,32,16",
            }
        )
        print(f"MLflow run ID: {run_id}")

    try:
        model, scaler, metadata = train_lstm(
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            sequence_length=sequence_length,
            max_rul=max_rul,
            tracker=tracker,
        )
        artifact_paths = save_artifacts(model, scaler, metadata)
        if tracker is not None:
            tracker.log_training_result(metadata, artifact_paths, model=model)
            tracker.end_run(status="FINISHED")
    except Exception:
        if tracker is not None:
            tracker.end_run(status="FAILED")
        raise
    return run_id


def main() -> None:
    """Train, save artifacts, and record the experiment in MLflow."""
    args = parse_args()
    run_tracked_training(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        sequence_length=args.sequence_length,
        max_rul=args.max_rul,
        tracking_uri=args.tracking_uri,
        experiment_name=args.experiment_name,
        run_name=args.run_name,
        enable_mlflow=not args.disable_mlflow,
    )


if __name__ == "__main__":
    main()

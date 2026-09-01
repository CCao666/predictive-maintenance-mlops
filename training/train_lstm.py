"""Train the upstream-equivalent PyTorch LSTM RUL model."""

import argparse
import json
import random
from dataclasses import asdict, dataclass
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
from training.mlflow_tracking import (
    DEFAULT_EXPERIMENT_NAME,
    DEFAULT_TRACKING_URI,
    MLflowExperimentTracker,
)
from training.model import ReferenceLSTMRegressor
from training.preprocessing import fit_feature_scaler, transform_features
from training.split import split_by_engine
from training.torch_dataset import RULSequenceDataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_PATH = PROJECT_ROOT / "data/cmapss/train_FD001.txt"
ARTIFACTS_DIRECTORY = PROJECT_ROOT / "artifacts"


@dataclass(frozen=True)
class TrainingConfig:
    """Values that define one reproducible training run."""

    data_path: Path = DEFAULT_DATA_PATH
    sequence_length: int = 50
    max_rul: int = 125
    batch_size: int = 32
    epochs: int = 200
    learning_rate: float = 1e-3
    validation_size: float = 0.2
    random_state: int = 42
    early_stopping_patience: int = 20
    lr_patience: int = 10

    def mlflow_params(self) -> dict[str, object]:
        params = asdict(self)
        params.pop("data_path")
        params.update(
            {
                "scaler": "RobustScaler",
                "lstm_hidden_sizes": "128,64,32",
                "attention_units": 64,
                "dense_sizes": "64,32,16",
            }
        )
        return params


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def select_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def prepare_datasets(
    config: TrainingConfig = TrainingConfig(),
) -> tuple[RULSequenceDataset, RULSequenceDataset, RobustScaler]:
    """Load, split, scale, and sequence FD001 without data leakage."""
    dataframe = add_train_rul(
        load_cmapss(config.data_path),
        max_rul=config.max_rul,
    )
    train_df, validation_df = split_by_engine(
        dataframe,
        validation_size=config.validation_size,
        random_state=config.random_state,
    )

    scaler = fit_feature_scaler(train_df)
    train_df = transform_features(train_df, scaler)
    validation_df = transform_features(validation_df, scaler)

    X_train, y_train = create_sequences(train_df, config.sequence_length)
    X_validation, y_validation = create_sequences(
        validation_df,
        config.sequence_length,
    )
    return (
        RULSequenceDataset(X_train, y_train),
        RULSequenceDataset(X_validation, y_validation),
        scaler,
    )


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: Adam,
    device: torch.device,
) -> float:
    """Train for one epoch and return mean squared error."""
    model.train()
    total_loss = 0.0
    sample_count = 0

    for sequences, targets in loader:
        sequences, targets = sequences.to(device), targets.to(device)
        optimizer.zero_grad()
        loss = nn.functional.mse_loss(model(sequences), targets)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item() * len(targets)
        sample_count += len(targets)

    return total_loss / sample_count


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    """Calculate MAE and RMSE in RUL cycles."""
    model.eval()
    errors = []

    with torch.no_grad():
        for sequences, targets in loader:
            predictions = model(sequences.to(device)).cpu()
            errors.append(predictions - targets)

    errors = torch.cat(errors)
    return {
        "mae": errors.abs().mean().item(),
        "rmse": errors.square().mean().sqrt().item(),
    }


def train_lstm(
    config: TrainingConfig = TrainingConfig(),
    tracker: MLflowExperimentTracker | None = None,
) -> tuple[ReferenceLSTMRegressor, RobustScaler, dict[str, object]]:
    """Train the reference model and restore its best validation weights."""
    if config.epochs <= 0:
        raise ValueError("epochs must be greater than zero")

    set_random_seed(config.random_state)
    device = select_device()
    train_data, validation_data, scaler = prepare_datasets(config)
    train_loader = DataLoader(
        train_data,
        batch_size=config.batch_size,
        shuffle=True,
    )
    validation_loader = DataLoader(
        validation_data,
        batch_size=config.batch_size,
    )

    model = ReferenceLSTMRegressor(len(FEATURE_COLUMNS)).to(device)
    optimizer = Adam(model.parameters(), lr=config.learning_rate)
    scheduler = ReduceLROnPlateau(
        optimizer,
        factor=0.5,
        patience=config.lr_patience,
        min_lr=1e-5,
    )

    best_rmse = float("inf")
    best_epoch = 0
    best_state = None
    epochs_without_improvement = 0
    history = []

    print(f"Device: {device}")
    print(f"Training sequences: {len(train_data)}")
    print(f"Validation sequences: {len(validation_data)}")

    for epoch in range(1, config.epochs + 1):
        train_rmse = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
        ) ** 0.5
        validation = evaluate(model, validation_loader, device)
        scheduler.step(validation["rmse"])

        metrics = {
            "epoch": epoch,
            "train_rmse": train_rmse,
            "validation_mae": validation["mae"],
            "validation_rmse": validation["rmse"],
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        history.append(metrics)
        if tracker:
            tracker.log_epoch(metrics, step=epoch)

        print(
            f"Epoch {epoch:03d}/{config.epochs} | "
            f"train RMSE {train_rmse:.2f} | "
            f"validation MAE {validation['mae']:.2f} | "
            f"validation RMSE {validation['rmse']:.2f} | "
            f"lr {metrics['learning_rate']:.6f}"
        )

        if validation["rmse"] < best_rmse - 0.001:
            best_rmse = validation["rmse"]
            best_epoch = epoch
            epochs_without_improvement = 0
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= config.early_stopping_patience:
            print(f"Early stopping at epoch {epoch}; best epoch was {best_epoch}")
            break

    if best_state is None:
        raise RuntimeError("Training did not produce model weights")

    model.load_state_dict(best_state)
    model.cpu()
    metadata = build_metadata(config, model, history, best_epoch, best_rmse)
    return model, scaler, metadata


def build_metadata(
    config: TrainingConfig,
    model: ReferenceLSTMRegressor,
    history: list[dict[str, float]],
    best_epoch: int,
    best_rmse: float,
) -> dict[str, object]:
    return {
        "model_name": "reference_lstm_rul_predictor",
        "source_architecture": "upstream-reference TensorFlow model, ported to PyTorch",
        "model_config": model.get_config(),
        "preprocessing": {
            "scaler": "RobustScaler",
            "feature_columns": FEATURE_COLUMNS,
            "max_rul": config.max_rul,
            "sequence_length": config.sequence_length,
        },
        "training": {
            "batch_size": config.batch_size,
            "maximum_epochs": config.epochs,
            "completed_epochs": len(history),
            "learning_rate": config.learning_rate,
            "validation_size": config.validation_size,
            "random_state": config.random_state,
            "early_stopping_patience": config.early_stopping_patience,
            "lr_patience": config.lr_patience,
        },
        "best_epoch": best_epoch,
        "best_validation_rmse": best_rmse,
        "history": history,
    }


def save_artifacts(
    model: ReferenceLSTMRegressor,
    scaler: RobustScaler,
    metadata: dict[str, object],
    output_dir: Path = ARTIFACTS_DIRECTORY,
) -> dict[str, Path]:
    """Save the model, scaler, and metrics used by inference."""
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "model": output_dir / "reference_lstm_model.pt",
        "scaler": output_dir / "robust_scaler.joblib",
        "metadata": output_dir / "training_metrics.json",
    }

    torch.save(
        {
            "model_class": "ReferenceLSTMRegressor",
            "model_config": model.get_config(),
            "state_dict": model.state_dict(),
        },
        paths["model"],
    )
    joblib.dump(scaler, paths["scaler"])
    paths["metadata"].write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    for name, path in paths.items():
        print(f"Saved {name}: {path}")
    return paths


def run_tracked_training(
    config: TrainingConfig,
    *,
    tracking_uri: str,
    experiment_name: str,
    run_name: str | None,
    enable_mlflow: bool = True,
    tags: dict[str, str] | None = None,
) -> str | None:
    """Train once, optionally recording the run in MLflow."""
    tracker = None
    run_id = None

    if enable_mlflow:
        tracker = MLflowExperimentTracker(tracking_uri, experiment_name)
        run_id = tracker.start_run(
            run_name,
            {
                "model_type": "ReferenceLSTMRegressor",
                "task": "remaining_useful_life_regression",
                "dataset": "NASA_CMAPSS_FD001",
                "framework": "pytorch",
                **(tags or {}),
            },
        )
        tracker.log_params(config.mlflow_params())
        print(f"MLflow run ID: {run_id}")

    try:
        model, scaler, metadata = train_lstm(config, tracker)
        paths = save_artifacts(model, scaler, metadata)
        if tracker:
            tracker.log_training_result(metadata, paths, model)
            tracker.end_run()
    except Exception:
        if tracker:
            tracker.end_run("FAILED")
        raise

    return run_id


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--sequence-length", type=int, default=50)
    parser.add_argument("--max-rul", type=int, default=125)
    parser.add_argument("--experiment-name", default=DEFAULT_EXPERIMENT_NAME)
    parser.add_argument("--run-name")
    parser.add_argument("--tracking-uri", default=DEFAULT_TRACKING_URI)
    parser.add_argument("--disable-mlflow", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = TrainingConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        sequence_length=args.sequence_length,
        max_rul=args.max_rul,
    )
    run_tracked_training(
        config,
        tracking_uri=args.tracking_uri,
        experiment_name=args.experiment_name,
        run_name=args.run_name,
        enable_mlflow=not args.disable_mlflow,
    )


if __name__ == "__main__":
    main()

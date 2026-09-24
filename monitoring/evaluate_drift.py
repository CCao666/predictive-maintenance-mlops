"""Compare C-MAPSS FD001-FD004 against the FD001 training baseline."""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from monitoring.drift import detect_data_drift
from training.dataset import load_cmapss


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = PROJECT_ROOT / "data/cmapss"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "artifacts/drift"
DATASET_IDS = ("FD001", "FD002", "FD003", "FD004")


def evaluate_datasets(
    data_dir: str | Path = DEFAULT_DATA_DIR,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> dict:
    """Evaluate all test datasets and save detailed JSON reports."""
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    reference = load_cmapss(data_dir / "train_FD001.txt")
    reports = {}

    for dataset_id in DATASET_IDS:
        current = load_cmapss(data_dir / f"test_{dataset_id}.txt")
        report = detect_data_drift(reference, current)
        report["reference_dataset"] = "train_FD001"
        report["current_dataset"] = f"test_{dataset_id}"
        reports[dataset_id] = report

        report_path = output_dir / f"{dataset_id}_report.json"
        report_path.write_text(
            json.dumps(report, indent=2),
            encoding="utf-8",
        )

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "reference_dataset": "train_FD001",
        "datasets": {
            dataset_id: {
                key: report[key]
                for key in (
                    "severity",
                    "drift_detected",
                    "drifted_feature_count",
                    "drifted_feature_ratio",
                    "overall_drift_score",
                    "mean_ks_statistic",
                    "median_ks_statistic",
                    "drifted_features",
                )
            }
            for dataset_id, report in reports.items()
        },
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    summary = evaluate_datasets(args.data_dir, args.output_dir)
    for dataset_id, result in summary["datasets"].items():
        print(
            f"{dataset_id}: {result['severity']:8} | "
            f"drifted {result['drifted_feature_count']:2}/24 | "
            f"score {result['overall_drift_score']:.3f}"
        )
    print(f"Reports saved to: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()

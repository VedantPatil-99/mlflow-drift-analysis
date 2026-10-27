"""Reference-only MLflow logging example.

This file is intentionally not executed in the submitted environment because
the user requested that MLflow not be installed.
"""

from __future__ import annotations

import json
from pathlib import Path

import mlflow


ROOT = Path(__file__).resolve().parent
REPORT_PATH = ROOT / "results" / "drift-report.json"


def main() -> None:
    report = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
    mlflow.set_experiment("galaxy-frog-retrieval-drift")
    with mlflow.start_run(run_name="reference-vs-current-telemetry"):
        mlflow.log_params(
            {
                "seed": report["seed"],
                "records_per_window": report["records"]["reference"],
                "psi_bins": report["psi_bins"],
                "psi_threshold": report["significant_drift_threshold"],
            }
        )
        mlflow.log_metrics(
            {f"psi_{name}": value for name, value in report["psi_by_feature"].items()}
        )
        mlflow.log_metric(
            "fallback_rate_change",
            report["current_summary"]["fallback_rate"]
            - report["reference_summary"]["fallback_rate"],
        )
        mlflow.log_artifact(str(REPORT_PATH), artifact_path="drift")


if __name__ == "__main__":
    main()

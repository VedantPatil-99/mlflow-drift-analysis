# MLflow Data Drift Analysis

This repository contains a deterministic academic data-drift analysis for
Galaxy Frog retrieval telemetry. The analysis compares a 100-record reference
window with a 100-record current window and calculates Population Stability
Index (PSI) for four numeric features.

## Run the analysis

```powershell
python analysis.py
```

The command uses only Python's standard library and writes:

- `data/reference.csv`
- `data/current.csv`
- `results/drift-report.json`

## MLflow status

MLflow is deliberately not installed or executed for this submission. The file
`mlflow_example.py` is reference code showing how the generated metrics and
artifact could be logged with MLflow 3.16.1. It should be run only in an
environment where MLflow has been installed intentionally.

Reference installation command (not executed):

```powershell
python -m pip install -r requirements-mlflow.txt
```

## Drift interpretation

- PSI below 0.10: low change
- PSI from 0.10 to below 0.25: moderate change
- PSI of 0.25 or above: significant drift

The sample is synthetic and reproducible (`seed=49`). It is appropriate for an
academic demonstration and is not production telemetry.

## Files

- `analysis.py`: generates the datasets and computes PSI without third-party packages
- `mlflow_example.py`: unexecuted MLflow tracking example
- `requirements-mlflow.txt`: optional pinned MLflow dependency
- `data/`: generated CSV datasets
- `results/`: generated JSON report

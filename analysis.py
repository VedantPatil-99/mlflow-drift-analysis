"""Generate deterministic telemetry samples and calculate data drift.

This script intentionally uses only the Python standard library. MLflow is not
required to generate the dataset or the drift report.
"""

from __future__ import annotations

import csv
import json
import math
import random
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
SEED = 49
RECORDS_PER_WINDOW = 100
PSI_BINS = 10
SIGNIFICANT_DRIFT_THRESHOLD = 0.25

FEATURES = (
    "query_length_tokens",
    "top1_score",
    "response_latency_ms",
    "evidence_count",
)


def clipped_gauss(rng: random.Random, mean: float, std: float, low: float, high: float) -> float:
    return min(high, max(low, rng.gauss(mean, std)))


def generate_window(kind: str, count: int, rng: random.Random) -> list[dict[str, float | int]]:
    rows: list[dict[str, float | int]] = []
    for row_id in range(1, count + 1):
        if kind == "reference":
            query_length = round(clipped_gauss(rng, 15.0, 4.0, 4.0, 30.0))
            top1_score = clipped_gauss(rng, 0.78, 0.07, 0.45, 0.95)
            latency = clipped_gauss(rng, 820.0, 140.0, 450.0, 1300.0)
            evidence_count = round(clipped_gauss(rng, 4.2, 1.0, 1.0, 8.0))
            fallback_probability = 0.09
        else:
            query_length = round(clipped_gauss(rng, 23.0, 5.5, 6.0, 42.0))
            top1_score = clipped_gauss(rng, 0.64, 0.10, 0.30, 0.92)
            latency = clipped_gauss(rng, 1240.0, 230.0, 650.0, 2100.0)
            evidence_count = round(clipped_gauss(rng, 3.0, 1.2, 1.0, 8.0))
            fallback_probability = 0.24

        fallback = int(rng.random() < fallback_probability)
        grounded = int(top1_score >= 0.60 and evidence_count >= 2 and not fallback)
        rows.append(
            {
                "record_id": row_id,
                "window": kind,
                "query_length_tokens": int(query_length),
                "top1_score": round(top1_score, 4),
                "response_latency_ms": round(latency, 2),
                "evidence_count": int(evidence_count),
                "fallback_used": fallback,
                "grounded_result": grounded,
            }
        )
    return rows


def percentile(sorted_values: list[float], fraction: float) -> float:
    if not sorted_values:
        raise ValueError("Cannot calculate a percentile of an empty list")
    position = (len(sorted_values) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    weight = position - lower
    return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight


def reference_edges(values: list[float], bins: int) -> list[float]:
    ordered = sorted(values)
    edges = [float("-inf")]
    for index in range(1, bins):
        edges.append(percentile(ordered, index / bins))
    edges.append(float("inf"))
    return edges


def distribution(values: list[float], edges: list[float]) -> list[float]:
    counts = [0] * (len(edges) - 1)
    for value in values:
        for index in range(len(edges) - 1):
            if edges[index] <= value < edges[index + 1]:
                counts[index] += 1
                break
    epsilon = 1e-6
    total = len(values)
    return [max(count / total, epsilon) for count in counts]


def psi(reference: list[float], current: list[float], bins: int) -> float:
    edges = reference_edges(reference, bins)
    expected = distribution(reference, edges)
    actual = distribution(current, edges)
    return sum((a - e) * math.log(a / e) for e, a in zip(expected, actual))


def summarize(rows: list[dict[str, float | int]]) -> dict[str, float]:
    summary: dict[str, float] = {}
    for feature in FEATURES:
        values = [float(row[feature]) for row in rows]
        summary[f"{feature}_mean"] = round(statistics.mean(values), 4)
    summary["fallback_rate"] = round(statistics.mean(float(row["fallback_used"]) for row in rows), 4)
    summary["grounded_result_rate"] = round(
        statistics.mean(float(row["grounded_result"]) for row in rows), 4
    )
    return summary


def write_csv(path: Path, rows: list[dict[str, float | int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    rng = random.Random(SEED)
    reference = generate_window("reference", RECORDS_PER_WINDOW, rng)
    current = generate_window("current", RECORDS_PER_WINDOW, rng)

    write_csv(DATA_DIR / "reference.csv", reference)
    write_csv(DATA_DIR / "current.csv", current)

    psi_by_feature: dict[str, float] = {}
    for feature in FEATURES:
        expected = [float(row[feature]) for row in reference]
        actual = [float(row[feature]) for row in current]
        psi_by_feature[feature] = round(psi(expected, actual, PSI_BINS), 4)

    significant_features = [
        feature for feature, value in psi_by_feature.items() if value >= SIGNIFICANT_DRIFT_THRESHOLD
    ]
    report = {
        "dataset": "Galaxy Frog Retrieval Telemetry (synthetic academic sample)",
        "seed": SEED,
        "records": {"reference": len(reference), "current": len(current)},
        "method": "Population Stability Index (PSI)",
        "psi_bins": PSI_BINS,
        "significant_drift_threshold": SIGNIFICANT_DRIFT_THRESHOLD,
        "psi_by_feature": psi_by_feature,
        "significant_drift_features": significant_features,
        "reference_summary": summarize(reference),
        "current_summary": summarize(current),
        "overall_drift_detected": bool(significant_features),
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = RESULTS_DIR / "drift-report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

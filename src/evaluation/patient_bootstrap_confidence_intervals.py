
from pathlib import Path

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    BASE_DIR
    / "results"
    / "segmentation"
    / "final_evaluation"
    / "final_metrics_by_patient.csv"
)

OUTPUT_DIR = (
    BASE_DIR
    / "results"
    / "segmentation"
    / "statistical_analysis"
    / "patient_bootstrap"
)

N_BOOTSTRAPS = 10000
CONFIDENCE_LEVEL = 0.95
RANDOM_STATE = 42

METRICS = {
    "dice": "dice_mean",
    "iou": "iou_mean",
    "precision": "precision_mean",
    "recall": "recall_mean",
    "boundary_dice": "boundary_dice_mean"
}


def bootstrap_confidence_interval(values, rng):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if len(values) == 0:
        return {
            "patient_count": 0,
            "mean": np.nan,
            "std": np.nan,
            "median": np.nan,
            "ci_lower": np.nan,
            "ci_upper": np.nan
        }

    bootstrap_means = np.empty(N_BOOTSTRAPS)

    for iteration in range(N_BOOTSTRAPS):
        sampled = rng.choice(
            values,
            size=len(values),
            replace=True
        )
        bootstrap_means[iteration] = np.mean(sampled)

    alpha = 1.0 - CONFIDENCE_LEVEL

    lower = np.percentile(
        bootstrap_means,
        100 * alpha / 2
    )

    upper = np.percentile(
        bootstrap_means,
        100 * (1 - alpha / 2)
    )

    return {
        "patient_count": len(values),
        "mean": np.mean(values),
        "std": np.std(values, ddof=1) if len(values) > 1 else 0.0,
        "median": np.median(values),
        "ci_lower": lower,
        "ci_upper": upper
    }


def analyze_group(group, group_name, rng):
    results = []

    for metric_name, column_name in METRICS.items():
        if column_name not in group.columns:
            raise ValueError(
                f"Missing metric column: {column_name}"
            )

        statistics = bootstrap_confidence_interval(
            group[column_name].to_numpy(),
            rng
        )

        results.append({
            "group": group_name,
            "metric": metric_name,
            **statistics,
            "bootstrap_iterations": N_BOOTSTRAPS,
            "confidence_level": CONFIDENCE_LEVEL
        })

    return results


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading final patient-level test results...")

    if not INPUT_FILE.exists():
        raise FileNotFoundError(INPUT_FILE)

    df = pd.read_csv(INPUT_FILE)

    required_columns = [
        "patient_id",
        "label",
        *METRICS.values()
    ]

    missing = [
        column for column in required_columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    if df["patient_id"].duplicated().any():
        raise ValueError(
            "Duplicate patient IDs found in patient-level results."
        )

    if df[["patient_id", "label"]].isnull().any().any():
        raise ValueError(
            "Missing patient IDs or tumor labels detected."
        )

    rng = np.random.default_rng(RANDOM_STATE)

    print(f"Patients: {len(df)}")
    print(f"Bootstrap iterations: {N_BOOTSTRAPS}")
    print(f"Confidence level: {CONFIDENCE_LEVEL:.0%}")

    all_results = []

    print("\nAnalyzing overall test set...")

    all_results.extend(
        analyze_group(df, "overall", rng)
    )

    for tumor_class in sorted(df["label"].unique()):
        class_data = df[
            df["label"] == tumor_class
        ].copy()

        print(
            f"Analyzing {tumor_class}: "
            f"{len(class_data)} patients..."
        )

        all_results.extend(
            analyze_group(
                class_data,
                tumor_class,
                rng
            )
        )

    results = pd.DataFrame(all_results)

    overall_results = results[
        results["group"] == "overall"
    ]

    class_results = results[
        results["group"] != "overall"
    ]

    results.to_csv(
        OUTPUT_DIR / "bootstrap_confidence_intervals_all.csv",
        index=False
    )

    overall_results.to_csv(
        OUTPUT_DIR / "bootstrap_confidence_intervals_overall.csv",
        index=False
    )

    class_results.to_csv(
        OUTPUT_DIR / "bootstrap_confidence_intervals_by_class.csv",
        index=False
    )

    print("\nOVERALL PATIENT-LEVEL RESULTS")
    print(overall_results.to_string(index=False))

    print("\nRESULTS BY TUMOR CLASS")
    print(class_results.to_string(index=False))

    print("\nBootstrap analysis completed successfully.")
    print(f"Results saved in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()


from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_PATH = (
    PROJECT_ROOT
    / "results"
    / "segmentation"
    / "baseline_vs_augmented"
    / "patient_comparison.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "segmentation"
    / "statistical_comparison"
)

RANDOM_SEED = 42
BOOTSTRAP_ITERATIONS = 10000
CONFIDENCE_LEVEL = 0.95

METRICS = [
    "dice_mean",
    "iou_mean",
    "precision_mean",
    "recall_mean",
    "boundary_dice_mean"
]

PRIMARY_METRIC = "dice_mean"


def load_patient_data():
    if not INPUT_PATH.is_file():
        raise FileNotFoundError(
            f"Patient comparison file not found: {INPUT_PATH}"
        )

    data = pd.read_csv(INPUT_PATH)

    required_columns = {
        "patient_id",
        "label",
        "model",
        *METRICS
    }

    missing = required_columns - set(data.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    data["model"] = data["model"].str.lower().str.strip()

    data = data[
        data["model"].isin(["baseline", "augmented"])
    ].copy()

    if data.duplicated(["patient_id", "model"]).any():
        raise ValueError(
            "Duplicate patient/model records found."
        )

    return data


def create_paired_data(data, metric):
    pivot = data.pivot(
        index="patient_id",
        columns="model",
        values=metric
    )

    labels = (
        data[["patient_id", "label"]]
        .drop_duplicates()
        .set_index("patient_id")
    )

    paired = pivot.join(labels)

    paired = paired.dropna(
        subset=["baseline", "augmented"]
    )

    paired["difference"] = (
        paired["augmented"] - paired["baseline"]
    )

    return paired


def bootstrap_confidence_interval(
    differences,
    iterations=BOOTSTRAP_ITERATIONS
):
    differences = np.asarray(
        differences,
        dtype=np.float64
    )

    differences = differences[
        np.isfinite(differences)
    ]

    if len(differences) == 0:
        return np.nan, np.nan

    rng = np.random.default_rng(RANDOM_SEED)

    bootstrap_means = np.empty(iterations)

    for index in range(iterations):
        sample = rng.choice(
            differences,
            size=len(differences),
            replace=True
        )

        bootstrap_means[index] = np.mean(sample)

    alpha = 1.0 - CONFIDENCE_LEVEL

    lower = np.quantile(
        bootstrap_means,
        alpha / 2
    )

    upper = np.quantile(
        bootstrap_means,
        1 - alpha / 2
    )

    return float(lower), float(upper)


def calculate_wilcoxon(differences):
    differences = np.asarray(
        differences,
        dtype=np.float64
    )

    differences = differences[
        np.isfinite(differences)
    ]

    if len(differences) == 0:
        return np.nan

    if np.allclose(differences, 0):
        return 1.0

    result = wilcoxon(
        differences,
        alternative="two-sided",
        zero_method="wilcox",
        method="auto"
    )

    return float(result.pvalue)


def analyze_metric(data, metric, group_name):
    paired = create_paired_data(data, metric)

    differences = paired["difference"].to_numpy()

    baseline_mean = paired["baseline"].mean()
    augmented_mean = paired["augmented"].mean()

    mean_difference = differences.mean()

    ci_lower, ci_upper = bootstrap_confidence_interval(
        differences
    )

    p_value = calculate_wilcoxon(differences)

    return {
        "group": group_name,
        "metric": metric,
        "patient_count": len(paired),
        "baseline_mean": baseline_mean,
        "augmented_mean": augmented_mean,
        "mean_difference": mean_difference,
        "ci_95_lower": ci_lower,
        "ci_95_upper": ci_upper,
        "wilcoxon_p_value": p_value
    }


def analyze_overall(data):
    results = []

    for metric in METRICS:
        result = analyze_metric(
            data,
            metric,
            "overall"
        )

        results.append(result)

    results_df = pd.DataFrame(results)

    valid = results_df["wilcoxon_p_value"].notna()

    results_df["holm_adjusted_p"] = np.nan

    if valid.any():
        results_df.loc[valid, "holm_adjusted_p"] = (
            multipletests(
                results_df.loc[
                    valid,
                    "wilcoxon_p_value"
                ],
                method="holm"
            )[1]
        )

    return results_df


def analyze_by_class(data):
    results = []

    for label in sorted(data["label"].unique()):
        class_data = data[
            data["label"] == label
        ]

        for metric in METRICS:
            result = analyze_metric(
                class_data,
                metric,
                label
            )

            results.append(result)

    results_df = pd.DataFrame(results)

    dice_results = results_df[
        results_df["metric"] == PRIMARY_METRIC
    ].copy()

    valid = dice_results["wilcoxon_p_value"].notna()

    dice_results["holm_adjusted_p"] = np.nan

    if valid.any():
        dice_results.loc[valid, "holm_adjusted_p"] = (
            multipletests(
                dice_results.loc[
                    valid,
                    "wilcoxon_p_value"
                ],
                method="holm"
            )[1]
        )

    results_df = results_df.merge(
        dice_results[
            ["group", "holm_adjusted_p"]
        ],
        on="group",
        how="left"
    )

    return results_df


def save_patient_differences(data):
    paired = create_paired_data(
        data,
        PRIMARY_METRIC
    ).reset_index()

    paired = paired.rename(
        columns={
            "baseline": "baseline_dice",
            "augmented": "augmented_dice",
            "difference": "dice_improvement"
        }
    )

    output_path = OUTPUT_DIR / "patient_dice_differences.csv"

    paired.to_csv(
        output_path,
        index=False
    )

    return paired


def create_dice_plot(overall, by_class):
    class_results = by_class[
        by_class["metric"] == PRIMARY_METRIC
    ].copy()

    plot_data = pd.concat(
        [
            overall[
                overall["metric"] == PRIMARY_METRIC
            ].assign(group="Overall"),
            class_results.assign(
                group=class_results["group"].str.capitalize()
            )
        ],
        ignore_index=True
    )

    labels = plot_data["group"].tolist()
    differences = plot_data["mean_difference"].to_numpy()
    lower = plot_data["ci_95_lower"].to_numpy()
    upper = plot_data["ci_95_upper"].to_numpy()

    errors = np.vstack([
        differences - lower,
        upper - differences
    ])

    figure, axis = plt.subplots(figsize=(9, 5))

    positions = np.arange(len(labels))

    axis.errorbar(
        positions,
        differences,
        yerr=errors,
        fmt="o",
        capsize=6,
        markersize=8
    )

    axis.axhline(
        0,
        linestyle="--",
        linewidth=1
    )

    axis.set_xticks(positions)
    axis.set_xticklabels(labels)

    axis.set_title(
        "Patient-Level Dice Improvement: Augmented vs Baseline"
    )

    axis.set_ylabel("Mean Dice Difference")
    axis.set_xlabel("Patient Group")
    axis.grid(axis="y", alpha=0.3)

    figure.tight_layout()

    figure.savefig(
        OUTPUT_DIR / "patient_level_dice_improvement.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(figure)


def main():
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    print("\nPATIENT-LEVEL STATISTICAL COMPARISON")
    print("=" * 75)

    data = load_patient_data()

    print(f"Patient-model records: {len(data)}")
    print(
        f"Unique patients: "
        f"{data['patient_id'].nunique()}"
    )

    print(
        f"Classes: "
        f"{', '.join(sorted(data['label'].unique()))}"
    )

    overall = analyze_overall(data)
    by_class = analyze_by_class(data)

    patient_differences = save_patient_differences(
        data
    )

    overall.to_csv(
        OUTPUT_DIR / "overall_statistical_results.csv",
        index=False
    )

    by_class.to_csv(
        OUTPUT_DIR / "class_statistical_results.csv",
        index=False
    )

    create_dice_plot(
        overall,
        by_class
    )

    print("\nOVERALL PATIENT-LEVEL RESULTS")
    print("=" * 75)

    print(
        overall[
            [
                "metric",
                "patient_count",
                "baseline_mean",
                "augmented_mean",
                "mean_difference",
                "ci_95_lower",
                "ci_95_upper",
                "wilcoxon_p_value",
                "holm_adjusted_p"
            ]
        ].to_string(index=False)
    )

    print("\nRESULTS BY TUMOR CLASS")
    print("=" * 75)

    print(
        by_class[
            [
                "group",
                "metric",
                "patient_count",
                "mean_difference",
                "ci_95_lower",
                "ci_95_upper",
                "wilcoxon_p_value",
                "holm_adjusted_p"
            ]
        ].to_string(index=False)
    )

    print("\nPATIENT-LEVEL DICE IMPROVEMENT")
    print("=" * 75)

    print(
        patient_differences[
            [
                "patient_id",
                "label",
                "baseline_dice",
                "augmented_dice",
                "dice_improvement"
            ]
        ].to_string(index=False)
    )

    print("\nSAVED FILES")
    print("=" * 75)
    print(OUTPUT_DIR / "overall_statistical_results.csv")
    print(OUTPUT_DIR / "class_statistical_results.csv")
    print(OUTPUT_DIR / "patient_dice_differences.csv")
    print(OUTPUT_DIR / "patient_level_dice_improvement.png")

    print("\nSTATISTICAL COMPARISON COMPLETED")


if __name__ == "__main__":
    main()

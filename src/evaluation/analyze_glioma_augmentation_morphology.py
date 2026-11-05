
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import spearmanr


PROJECT_ROOT = Path(__file__).resolve().parents[2]

MORPHOLOGY_PATH = (
    PROJECT_ROOT
    / "results/segmentation/statistical_analysis/glioma_patient_morphology.csv"
)

DICE_DIFFERENCES_PATH = (
    PROJECT_ROOT
    / "results/segmentation/statistical_comparison/patient_dice_differences.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results/segmentation/statistical_analysis/augmentation_morphology"
)

FEATURES = [
    "tumor_area_percent",
    "perimeter_to_area",
    "compactness",
    "eccentricity"
]


def holm_correction(p_values):
    p_values = np.asarray(p_values, dtype=float)
    count = len(p_values)

    order = np.argsort(p_values)
    sorted_p = p_values[order]

    adjusted = np.maximum.accumulate(
        (count - np.arange(count)) * sorted_p
    )

    adjusted = np.minimum(adjusted, 1.0)

    result = np.empty(count)
    result[order] = adjusted

    return result


def load_and_merge_data():
    morphology = pd.read_csv(MORPHOLOGY_PATH)
    dice_differences = pd.read_csv(DICE_DIFFERENCES_PATH)

    glioma_dice = dice_differences[
        dice_differences["label"] == "glioma"
    ].copy()

    merged = morphology.merge(
        glioma_dice[
            [
                "patient_id",
                "baseline_dice",
                "augmented_dice",
                "dice_improvement"
            ]
        ],
        on="patient_id",
        how="inner",
        validate="one_to_one"
    )

    if len(merged) != len(morphology):
        print(
            "Warning: Some morphology patients did not match "
            "the Dice comparison file."
        )

    if len(merged) < 3:
        raise ValueError(
            "Not enough matched glioma patients for correlation analysis."
        )

    return merged


def calculate_correlations(data):
    results = []

    for feature in FEATURES:
        valid = data[[feature, "dice_improvement"]].dropna()

        if valid[feature].nunique() < 2:
            print(f"Skipping constant feature: {feature}")
            continue

        rho, p_value = spearmanr(
            valid[feature],
            valid["dice_improvement"]
        )

        results.append({
            "feature": feature,
            "patient_count": len(valid),
            "spearman_rho": rho,
            "p_value": p_value
        })

    results_df = pd.DataFrame(results)

    if not results_df.empty:
        results_df["holm_adjusted_p"] = holm_correction(
            results_df["p_value"]
        )

    return results_df


def create_scatter_plots(data):
    figure, axes = plt.subplots(
        2,
        2,
        figsize=(12, 10)
    )

    for axis, feature in zip(axes.flat, FEATURES):
        axis.scatter(
            data[feature],
            data["dice_improvement"],
            s=55,
            alpha=0.8
        )

        for _, row in data.iterrows():
            axis.annotate(
                str(row["patient_id"]),
                (row[feature], row["dice_improvement"]),
                fontsize=7,
                alpha=0.7
            )

        axis.axhline(0, linestyle="--", linewidth=1)
        axis.set_title(feature.replace("_", " ").title())
        axis.set_xlabel(feature.replace("_", " ").title())
        axis.set_ylabel("Dice Improvement")
        axis.grid(alpha=0.3)

    figure.suptitle(
        "Glioma Morphology vs. Augmentation-Related Dice Improvement",
        fontsize=13
    )

    figure.tight_layout()

    figure.savefig(
        OUTPUT_DIR / "glioma_morphology_vs_dice_improvement.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(figure)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("\nGLIOMA MORPHOLOGY AND AUGMENTATION ANALYSIS")
    print("=" * 70)

    data = load_and_merge_data()

    print(f"Matched glioma patients: {len(data)}")

    print("\nPATIENT-LEVEL DATA")
    print("=" * 70)

    display_columns = [
        "patient_id",
        "tumor_area_percent",
        "perimeter_to_area",
        "compactness",
        "eccentricity",
        "baseline_dice",
        "augmented_dice",
        "dice_improvement"
    ]

    print(
        data[display_columns].to_string(index=False)
    )

    print("\nMORPHOLOGY VS. DICE IMPROVEMENT")
    print("=" * 70)

    correlations = calculate_correlations(data)

    if correlations.empty:
        raise ValueError("No valid morphology correlations were calculated.")

    print(
        correlations.to_string(index=False)
    )

    correlations.to_csv(
        OUTPUT_DIR / "glioma_augmentation_morphology_correlations.csv",
        index=False
    )

    data.to_csv(
        OUTPUT_DIR / "glioma_augmentation_morphology_merged.csv",
        index=False
    )

    create_scatter_plots(data)

    print("\nSAVED FILES")
    print("=" * 70)
    print(OUTPUT_DIR / "glioma_augmentation_morphology_correlations.csv")
    print(OUTPUT_DIR / "glioma_augmentation_morphology_merged.csv")
    print(OUTPUT_DIR / "glioma_morphology_vs_dice_improvement.png")

    print("\nANALYSIS COMPLETED")


if __name__ == "__main__":
    main()

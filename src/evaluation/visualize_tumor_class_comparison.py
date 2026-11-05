
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


BASE_DIR = Path(__file__).resolve().parents[2]

PATIENT_FILE = (
    BASE_DIR
    / "results"
    / "segmentation"
    / "final_evaluation"
    / "final_metrics_by_patient.csv"
)

COMPARISON_FILE = (
    BASE_DIR
    / "results"
    / "segmentation"
    / "statistical_analysis"
    / "tumor_class_comparison"
    / "pairwise_class_comparisons.csv"
)

OUTPUT_DIR = (
    BASE_DIR
    / "results"
    / "segmentation"
    / "statistical_analysis"
    / "tumor_class_comparison"
    / "figures"
)

CLASS_ORDER = ["glioma", "meningioma", "pituitary"]

CLASS_LABELS = {
    "glioma": "Glioma",
    "meningioma": "Meningioma",
    "pituitary": "Pituitary"
}


def load_data():
    if not PATIENT_FILE.exists():
        raise FileNotFoundError(PATIENT_FILE)

    if not COMPARISON_FILE.exists():
        raise FileNotFoundError(COMPARISON_FILE)

    data = pd.read_csv(PATIENT_FILE)
    comparisons = pd.read_csv(COMPARISON_FILE)

    required_columns = {"patient_id", "label", "dice_mean"}

    if not required_columns.issubset(data.columns):
        raise ValueError("Patient CSV is missing required columns.")

    data = data[data["label"].isin(CLASS_ORDER)].copy()
    data["dice_mean"] = pd.to_numeric(
        data["dice_mean"], errors="coerce"
    )
    data = data.dropna(subset=["dice_mean"])

    if data.empty:
        raise ValueError("No valid patient-level Dice values found.")

    return data, comparisons


def add_significance_bracket(
    ax,
    x1,
    x2,
    y,
    height,
    p_value
):
    if p_value < 0.001:
        significance = "***"
    elif p_value < 0.01:
        significance = "**"
    elif p_value < 0.05:
        significance = "*"
    else:
        significance = "ns"

    ax.plot(
        [x1, x1, x2, x2],
        [y, y + height, y + height, y],
        linewidth=1.2
    )

    ax.text(
        (x1 + x2) / 2,
        y + height,
        f"{significance}  (p={p_value:.4f})",
        ha="center",
        va="bottom",
        fontsize=9
    )


def create_boxplot(data, comparisons):
    values = [
        data.loc[
            data["label"] == label, "dice_mean"
        ].to_numpy(dtype=float)
        for label in CLASS_ORDER
    ]

    if any(len(group) == 0 for group in values):
        raise ValueError("One or more tumor classes have no observations.")

    fig, ax = plt.subplots(figsize=(10, 7))

    positions = [1, 2, 3]

    boxplot = ax.boxplot(
        values,
        positions=positions,
        widths=0.55,
        patch_artist=True,
        showmeans=True,
        meanprops={
            "marker": "D",
            "markerfacecolor": "white",
            "markeredgecolor": "black",
            "markersize": 6
        },
        medianprops={
            "linewidth": 2
        },
        whiskerprops={
            "linewidth": 1.3
        },
        capprops={
            "linewidth": 1.3
        },
        flierprops={
            "marker": "o",
            "markersize": 4,
            "alpha": 0.5
        }
    )

    for patch in boxplot["boxes"]:
        patch.set_alpha(0.65)

    rng = np.random.default_rng(42)

    for position, group in zip(positions, values):
        jitter = rng.uniform(-0.09, 0.09, size=len(group))

        ax.scatter(
            position + jitter,
            group,
            s=35,
            alpha=0.75,
            edgecolors="black",
            linewidths=0.4,
            zorder=3
        )

    ax.set_xticks(positions)
    ax.set_xticklabels(
        [CLASS_LABELS[label] for label in CLASS_ORDER],
        fontsize=11
    )

    ax.set_ylabel("Patient-level Mean Dice Coefficient", fontsize=12)
    ax.set_xlabel("Brain Tumor Class", fontsize=12)
    ax.set_title(
        "Patient-Level U-Net Segmentation Performance by Tumor Class",
        fontsize=14,
        pad=20
    )

    ax.set_ylim(0, 1.28)
    ax.set_yticks(np.arange(0, 1.01, 0.1))
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.set_axisbelow(True)

    comparison_positions = {
        "glioma vs meningioma": (1, 2),
        "glioma vs pituitary": (1, 3),
        "meningioma vs pituitary": (2, 3)
    }

    bracket_levels = {
        "glioma vs meningioma": 1.03,
        "meningioma vs pituitary": 1.12,
        "glioma vs pituitary": 1.21
    }

    for _, row in comparisons.iterrows():
        comparison = row["comparison"].lower()

        if comparison not in comparison_positions:
            continue

        x1, x2 = comparison_positions[comparison]
        y = bracket_levels[comparison]

        add_significance_bracket(
            ax,
            x1,
            x2,
            y,
            0.025,
            float(row["p_value_holm"])
        )

    ax.text(
        0.99,
        0.015,
        "Holm-adjusted p-values: * p < 0.05, "
        "** p < 0.01, *** p < 0.001",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=8
    )

    fig.tight_layout()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    png_path = OUTPUT_DIR / "tumor_class_dice_comparison.png"
    pdf_path = OUTPUT_DIR / "tumor_class_dice_comparison.pdf"

    fig.savefig(
        png_path,
        dpi=600,
        bbox_inches="tight"
    )

    fig.savefig(
        pdf_path,
        bbox_inches="tight"
    )

    plt.close(fig)

    print("\nVISUALIZATION COMPLETED")
    print(f"PNG saved: {png_path}")
    print(f"PDF saved: {pdf_path}")


def main():
    print("Loading patient-level results...")

    data, comparisons = load_data()

    print(f"Patients included: {len(data)}")

    print("\nPatients per class:")

    print(
        data.groupby("label")["patient_id"]
        .nunique()
        .reindex(CLASS_ORDER)
        .to_string()
    )

    print("\nGenerating comparison figure...")

    create_boxplot(data, comparisons)


if __name__ == "__main__":
    main()

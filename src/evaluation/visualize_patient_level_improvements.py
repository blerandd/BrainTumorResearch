
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_PATH = (
    PROJECT_ROOT
    / "results"
    / "segmentation"
    / "baseline_vs_augmented"
    / "statistical_analysis"
    / "patient_level_paired_results.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "segmentation"
    / "baseline_vs_augmented"
    / "statistical_analysis"
)

OUTPUT_PATH = OUTPUT_DIR / "patient_level_improvement_analysis.png"

CLASS_COLORS = {
    "glioma": "#2878B5",
    "meningioma": "#E07A32",
    "pituitary": "#31966B"
}


def load_data():
    if not INPUT_PATH.is_file():
        raise FileNotFoundError(
            f"Input file not found: {INPUT_PATH}"
        )

    data = pd.read_csv(INPUT_PATH)

    required_columns = {
        "patient_id",
        "label",
        "baseline_dice",
        "augmented_dice",
        "dice_change"
    }

    missing = required_columns - set(data.columns)

    if missing:
        raise ValueError(
            f"Missing columns: {sorted(missing)}"
        )

    data = data.sort_values(
        "dice_change"
    ).reset_index(drop=True)

    return data


def plot_patient_improvements(axis, data):
    positions = np.arange(len(data))

    for position, row in data.iterrows():
        color = CLASS_COLORS.get(
            row["label"],
            "gray"
        )

        axis.plot(
            [0, 1],
            [
                row["baseline_dice"],
                row["augmented_dice"]
            ],
            color=color,
            alpha=0.65,
            linewidth=1.5
        )

        axis.scatter(
            0,
            row["baseline_dice"],
            color=color,
            s=22,
            zorder=3
        )

        axis.scatter(
            1,
            row["augmented_dice"],
            color=color,
            s=22,
            zorder=3
        )

    axis.set_xlim(-0.25, 1.25)
    axis.set_xticks([0, 1])
    axis.set_xticklabels(
        ["Baseline U-Net", "Augmented U-Net"]
    )

    axis.set_ylabel("Patient-level Mean Dice")
    axis.set_title("Baseline vs Augmented Dice per Patient")
    axis.set_ylim(0, 1)
    axis.grid(axis="y", alpha=0.25)


def plot_change_by_patient(axis, data):
    positions = np.arange(len(data))

    colors = [
        CLASS_COLORS.get(label, "gray")
        for label in data["label"]
    ]

    axis.barh(
        positions,
        data["dice_change"],
        color=colors,
        alpha=0.85
    )

    axis.axvline(
        0,
        color="black",
        linewidth=1
    )

    axis.set_yticks(positions)
    axis.set_yticklabels(
        data["patient_id"],
        fontsize=7
    )

    axis.set_xlabel("Augmented Dice - Baseline Dice")
    axis.set_ylabel("Patient")
    axis.set_title("Patient-Level Dice Improvement")
    axis.grid(axis="x", alpha=0.25)

    axis.invert_yaxis()


def create_figure(data):
    figure, axes = plt.subplots(
        1,
        2,
        figsize=(19, 12),
        gridspec_kw={"width_ratios": [1, 1.3]}
    )

    plot_patient_improvements(
        axes[0],
        data
    )

    plot_change_by_patient(
        axes[1],
        data
    )

    legend_handles = []

    for label, color in CLASS_COLORS.items():
        legend_handles.append(
            plt.Line2D(
                [0],
                [0],
                marker="o",
                color=color,
                label=label.capitalize(),
                markersize=7,
                linestyle=""
            )
        )

    figure.legend(
        handles=legend_handles,
        loc="upper center",
        ncol=3,
        title="Tumor Class"
    )

    figure.suptitle(
        "Patient-Level Baseline vs Augmented U-Net Analysis",
        fontsize=16,
        y=0.98
    )

    figure.tight_layout(
        rect=[0, 0, 1, 0.95]
    )

    figure.savefig(
        OUTPUT_PATH,
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()
    plt.close(figure)


def print_summary(data):
    improved = int((data["dice_change"] > 0).sum())
    deteriorated = int((data["dice_change"] < 0).sum())
    unchanged = int((data["dice_change"] == 0).sum())

    print("\nPATIENT-LEVEL VISUALIZATION SUMMARY")
    print("=" * 65)

    print(f"Total patients: {len(data)}")
    print(f"Improved: {improved}")
    print(f"Deteriorated: {deteriorated}")
    print(f"Unchanged: {unchanged}")

    print(
        f"Mean Dice change: "
        f"{data['dice_change'].mean():.6f}"
    )

    print(
        f"Median Dice change: "
        f"{data['dice_change'].median():.6f}"
    )

    print("\nLARGEST IMPROVEMENTS")
    print(
        data.nlargest(
            5,
            "dice_change"
        )[
            ["patient_id", "label", "dice_change"]
        ].to_string(index=False)
    )

    print("\nLARGEST DETERIORATIONS")
    print(
        data.nsmallest(
            5,
            "dice_change"
        )[
            ["patient_id", "label", "dice_change"]
        ].to_string(index=False)
    )


def main():
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    print("\nLOADING PATIENT-LEVEL RESULTS")
    print("=" * 65)

    data = load_data()

    print_summary(data)

    print("\nCREATING VISUALIZATION")
    print("=" * 65)

    create_figure(data)

    print(f"\nVisualization saved to:\n{OUTPUT_PATH}")
    print("\nPATIENT-LEVEL VISUALIZATION COMPLETED")


if __name__ == "__main__":
    main()


from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

RESULTS_DIR = (
    PROJECT_ROOT
    / "results"
    / "segmentation"
    / "three_model_comparison"
    / "statistical_analysis"
)

OUTPUT_DIR = RESULTS_DIR / "visualizations"

MODEL_ORDER = [
    "baseline",
    "flip_only",
    "flip_rotation"
]

MODEL_LABELS = {
    "baseline": "Baseline U-Net",
    "flip_only": "Flip-only U-Net",
    "flip_rotation": "Flip + Rotation U-Net"
}

MODEL_COLORS = {
    "baseline": "#64748b",
    "flip_only": "#2563eb",
    "flip_rotation": "#16a34a"
}


def load_results():
    wide = pd.read_csv(
        RESULTS_DIR / "patient_level_wide_results.csv"
    )

    summary = pd.read_csv(
        RESULTS_DIR / "patient_level_model_summary.csv"
    )

    friedman = pd.read_csv(
        RESULTS_DIR / "friedman_test_results.csv"
    )

    pairwise = pd.read_csv(
        RESULTS_DIR / "pairwise_wilcoxon_results.csv"
    )

    return wide, summary, friedman, pairwise


def create_output_directory():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def plot_overall_distributions(axis, wide):
    data = [
        wide[model].dropna().to_numpy()
        for model in MODEL_ORDER
    ]

    box = axis.boxplot(
        data,
        patch_artist=True,
        widths=0.55,
        showmeans=True,
        meanline=True
    )

    for patch, model in zip(box["boxes"], MODEL_ORDER):
        patch.set_facecolor(MODEL_COLORS[model])
        patch.set_alpha(0.65)

    for median in box["medians"]:
        median.set_color("black")
        median.set_linewidth(1.5)

    for index, model in enumerate(MODEL_ORDER, start=1):
        values = wide[model].dropna().to_numpy()

        jitter = np.random.default_rng(42).normal(
            0,
            0.045,
            size=len(values)
        )

        axis.scatter(
            index + jitter,
            values,
            color=MODEL_COLORS[model],
            alpha=0.45,
            s=18,
            edgecolors="none"
        )

    axis.set_xticks(range(1, len(MODEL_ORDER) + 1))

    axis.set_xticklabels([
        MODEL_LABELS[model]
        for model in MODEL_ORDER
    ])

    axis.set_ylabel("Patient-level Dice")
    axis.set_title("A. Patient-level Dice distributions")
    axis.set_ylim(0, 1)
    axis.grid(axis="y", alpha=0.25)


def plot_paired_patient_scores(axis, wide):
    x_positions = np.arange(len(MODEL_ORDER))

    for _, patient in wide.iterrows():
        values = [
            patient[model]
            for model in MODEL_ORDER
        ]

        axis.plot(
            x_positions,
            values,
            color="#94a3b8",
            alpha=0.35,
            linewidth=0.9
        )

    for index, model in enumerate(MODEL_ORDER):
        axis.scatter(
            np.full(len(wide), index),
            wide[model],
            color=MODEL_COLORS[model],
            alpha=0.65,
            s=17
        )

    means = [
        wide[model].mean()
        for model in MODEL_ORDER
    ]

    axis.plot(
        x_positions,
        means,
        color="black",
        marker="D",
        linewidth=2.5,
        markersize=7,
        label="Mean Dice"
    )

    axis.set_xticks(x_positions)

    axis.set_xticklabels([
        MODEL_LABELS[model]
        for model in MODEL_ORDER
    ])

    axis.set_ylabel("Patient-level Dice")
    axis.set_title("B. Paired patient-level performance")
    axis.set_ylim(0, 1)
    axis.grid(axis="y", alpha=0.25)
    axis.legend()


def plot_class_distributions(axis, wide):
    classes = [
        "glioma",
        "meningioma",
        "pituitary"
    ]

    positions = []
    data = []
    colors = []

    position = 1

    for class_name in classes:
        class_data = wide[
            wide["label"] == class_name
        ]

        for model in MODEL_ORDER:
            data.append(
                class_data[model].dropna().to_numpy()
            )

            positions.append(position)
            colors.append(MODEL_COLORS[model])

            position += 1

        position += 1

    box = axis.boxplot(
        data,
        positions=positions,
        widths=0.55,
        patch_artist=True,
        showmeans=True
    )

    for patch, color in zip(box["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    class_centers = [2, 6, 10]

    axis.set_xticks(class_centers)

    axis.set_xticklabels([
        "Glioma",
        "Meningioma",
        "Pituitary"
    ])

    axis.set_ylabel("Patient-level Dice")
    axis.set_title("C. Performance by tumor class")
    axis.set_ylim(0, 1)
    axis.grid(axis="y", alpha=0.25)

    from matplotlib.patches import Patch

    legend_handles = [
        Patch(
            facecolor=MODEL_COLORS[model],
            label=MODEL_LABELS[model]
        )
        for model in MODEL_ORDER
    ]

    axis.legend(handles=legend_handles, fontsize=8)


def plot_pairwise_comparisons(axis, pairwise):
    overall = pairwise[
        pairwise["analysis"] == "overall"
    ].copy()

    overall = overall.reset_index(drop=True)

    y_positions = np.arange(len(overall))

    differences = overall[
        "mean_dice_difference"
    ].to_numpy()

    lower = overall[
        "bootstrap_95_ci_lower"
    ].to_numpy()

    upper = overall[
        "bootstrap_95_ci_upper"
    ].to_numpy()

    x_errors = np.vstack([
        differences - lower,
        upper - differences
    ])

    axis.errorbar(
        differences,
        y_positions,
        xerr=x_errors,
        fmt="o",
        color="#2563eb",
        ecolor="#64748b",
        capsize=5,
        markersize=7,
        linewidth=1.5
    )

    axis.axvline(
        0,
        color="black",
        linestyle="--",
        linewidth=1
    )

    labels = []

    for _, row in overall.iterrows():
        comparison = row["comparison"]

        adjusted_p = row["holm_adjusted_p_value"]

        significant = row["significant_after_holm"]

        status = (
            "Significant"
            if significant
            else "Not significant"
        )

        labels.append(
            f"{comparison}\n"
            f"Holm p={adjusted_p:.4f} ({status})"
        )

    axis.set_yticks(y_positions)
    axis.set_yticklabels(labels, fontsize=8)

    axis.set_xlabel("Mean patient Dice difference")
    axis.set_title("D. Pairwise comparisons with 95% bootstrap CI")

    axis.grid(axis="x", alpha=0.25)


def create_figure(wide, friedman, pairwise):
    figure, axes = plt.subplots(
        2,
        2,
        figsize=(17, 12)
    )

    plot_overall_distributions(
        axes[0, 0],
        wide
    )

    plot_paired_patient_scores(
        axes[0, 1],
        wide
    )

    plot_class_distributions(
        axes[1, 0],
        wide
    )

    plot_pairwise_comparisons(
        axes[1, 1],
        pairwise
    )

    overall_friedman = friedman[
        friedman["analysis"] == "overall"
    ].iloc[0]

    figure.suptitle(
        "Patient-Level Statistical Comparison of Three U-Net Models\n"
        f"Friedman test: statistic = "
        f"{overall_friedman['friedman_statistic']:.3f}, "
        f"p = {overall_friedman['friedman_p_value']:.4f}",
        fontsize=15,
        fontweight="bold"
    )

    figure.tight_layout(
        rect=[0, 0, 1, 0.93]
    )

    output_path = (
        OUTPUT_DIR
        / "three_model_patient_level_statistical_analysis.png"
    )

    figure.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()
    plt.close(figure)

    print("\nVISUALIZATION SAVED")
    print("=" * 65)
    print(output_path)


def main():
    print("\nTHREE-MODEL STATISTICAL VISUALIZATION")
    print("=" * 65)

    create_output_directory()

    wide, summary, friedman, pairwise = load_results()

    print(f"Patients analyzed: {len(wide)}")
    print(f"Models: {', '.join(MODEL_ORDER)}")

    print("\nMODEL SUMMARY")
    print(summary.to_string(index=False))

    print("\nOVERALL FRIEDMAN TEST")
    print(
        friedman[
            friedman["analysis"] == "overall"
        ].to_string(index=False)
    )

    print("\nOVERALL PAIRWISE COMPARISONS")
    print(
        pairwise[
            pairwise["analysis"] == "overall"
        ].to_string(index=False)
    )

    create_figure(
        wide,
        friedman,
        pairwise
    )

    print("\nSTATISTICAL VISUALIZATION COMPLETED")


if __name__ == "__main__":
    main()

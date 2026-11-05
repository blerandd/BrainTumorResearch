
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]

SEGMENTATION_FILE = (
    ROOT / "results" / "segmentation" /
    "patient_segmentation_metrics.csv"
)

INTENSITY_FILE = (
    ROOT / "results" / "segmentation" / "diagnostics" /
    "patient_intensity_contrast.csv"
)

OUTPUT_DIR = (
    ROOT / "results" / "segmentation" / "statistical_analysis"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

segmentation = pd.read_csv(SEGMENTATION_FILE)
intensity = pd.read_csv(INTENSITY_FILE)

segmentation["patient_id"] = (
    segmentation["patient_id"].astype(str).str.strip()
)

intensity["patient_id"] = (
    intensity["patient_id"].astype(str).str.strip()
)

intensity = intensity[intensity["split"] == "test"].copy()

merged = pd.merge(
    segmentation,
    intensity[
        [
            "patient_id",
            "label",
            "intensity_difference",
            "intensity_ratio"
        ]
    ],
    on="patient_id",
    how="inner",
    validate="one_to_one"
)

if merged.empty:
    raise ValueError("No matching test patients were found.")

if merged["patient_id"].nunique() != len(merged):
    raise ValueError("Duplicate patient records detected.")

merged.to_csv(
    OUTPUT_DIR / "patient_intensity_dice_merged.csv",
    index=False
)


def analyze_relationship(data, title):
    print(f"\n{title}")
    print("=" * 65)
    print(f"Number of patients: {len(data)}")

    if len(data) < 3:
        print("Not enough patients for correlation analysis.")
        return None

    result = spearmanr(
        data["intensity_difference"],
        data["dice_mean"]
    )

    print(f"Spearman correlation: {result.statistic:.4f}")
    print(f"P-value: {result.pvalue:.6f}")

    return result


overall_result = analyze_relationship(
    merged,
    "ALL TEST PATIENTS"
)

glioma = merged[merged["label"] == "glioma"].copy()

glioma_result = analyze_relationship(
    glioma,
    "GLIOMA TEST PATIENTS"
)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

for label, group in merged.groupby("label"):
    axes[0].scatter(
        group["intensity_difference"],
        group["dice_mean"],
        label=label,
        alpha=0.8
    )

axes[0].set_title("Intensity Contrast vs Dice — All Classes")
axes[0].set_xlabel("Tumor-to-background intensity difference")
axes[0].set_ylabel("Patient-level Dice")
axes[0].legend()
axes[0].grid(alpha=0.3)

axes[1].scatter(
    glioma["intensity_difference"],
    glioma["dice_mean"],
    alpha=0.8
)

for row in glioma.itertuples():
    axes[1].annotate(
        str(row.patient_id),
        (
            row.intensity_difference,
            row.dice_mean
        ),
        fontsize=7
    )

axes[1].set_title("Intensity Contrast vs Dice — Glioma Only")
axes[1].set_xlabel("Tumor-to-background intensity difference")
axes[1].set_ylabel("Patient-level Dice")
axes[1].grid(alpha=0.3)

fig.suptitle(
    "Association Between MRI Intensity Contrast and U-Net Performance",
    fontsize=14
)

plt.tight_layout()

plot_path = OUTPUT_DIR / "intensity_dice_relationship.png"
plt.savefig(plot_path, dpi=300, bbox_inches="tight")
plt.show()

print("\nSAVED RESULTS")
print(OUTPUT_DIR / "patient_intensity_dice_merged.csv")
print(plot_path)

print("\nANALYSIS COMPLETED")

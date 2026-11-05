
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "processed"
FIGURE_DIR = ROOT / "results" / "figures"

FIGURE_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA_DIR / "splits.csv")

print("\nDATASET CLASS DISTRIBUTION")
print(df["label"].value_counts())

print("\nIMAGES PER PATIENT")
patient_counts = df.groupby("patient_id").size()
print(patient_counts.describe())

print("\nTUMOR AREA ANALYSIS")

areas = []

for row in df.itertuples():
    mask_path = DATA_DIR / row.mask_path
    mask = np.asarray(Image.open(mask_path)) > 0
    tumor_area = mask.sum() / mask.size * 100

    areas.append({
        "label": row.label,
        "patient_id": row.patient_id,
        "split": row.split,
        "tumor_area_percent": tumor_area
    })

area_df = pd.DataFrame(areas)

print(area_df.groupby("label")["tumor_area_percent"].describe())

fig, axes = plt.subplots(2, 2, figsize=(15, 11))

class_counts = df["label"].value_counts()
axes[0, 0].bar(class_counts.index, class_counts.values)
axes[0, 0].set_title("MRI Images per Tumor Class")
axes[0, 0].set_ylabel("Number of images")

axes[0, 1].hist(patient_counts.values, bins=20, edgecolor="black")
axes[0, 1].set_title("MRI Images per Patient")
axes[0, 1].set_xlabel("Number of images")
axes[0, 1].set_ylabel("Number of patients")

for label in sorted(area_df["label"].unique()):
    values = area_df.loc[
        area_df["label"] == label,
        "tumor_area_percent"
    ]
    axes[1, 0].hist(values, bins=30, alpha=0.5, label=label)

axes[1, 0].set_title("Tumor Area Distribution")
axes[1, 0].set_xlabel("Tumor area (% of image)")
axes[1, 0].set_ylabel("Number of images")
axes[1, 0].legend()

axes[1, 1].boxplot(
    [
        area_df.loc[
            area_df["label"] == label,
            "tumor_area_percent"
        ]
        for label in sorted(area_df["label"].unique())
    ],
    tick_labels=sorted(area_df["label"].unique())
)
axes[1, 1].set_title("Tumor Area by Class")
axes[1, 1].set_ylabel("Tumor area (% of image)")

fig.suptitle(
    "Exploratory Analysis of the Figshare Brain Tumor Dataset",
    fontsize=16,
    fontweight="bold"
)

plt.tight_layout()

output_path = FIGURE_DIR / "dataset_eda.png"
plt.savefig(output_path, dpi=300, bbox_inches="tight")
plt.show()

area_df.to_csv(
    DATA_DIR / "tumor_area_analysis.csv",
    index=False
)

print("\nEDA COMPLETED")
print(f"Figure saved to: {output_path}")
print("Tumor area statistics saved.")

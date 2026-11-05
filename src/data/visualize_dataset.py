
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
train_df = df[df["split"] == "train"]

classes = ["glioma", "meningioma", "pituitary"]

fig, axes = plt.subplots(3, 3, figsize=(15, 16))

for row_index, class_name in enumerate(classes):
    sample = train_df[
        train_df["label"] == class_name
    ].sample(n=1, random_state=42).iloc[0]

    image_path = DATA_DIR / sample["image_path"]
    mask_path = DATA_DIR / sample["mask_path"]

    image = np.asarray(Image.open(image_path)).astype(np.float32)
    mask = np.asarray(Image.open(mask_path)) > 0

    low, high = np.percentile(image, [1, 99])
    image = np.clip((image - low) / (high - low + 1e-8), 0, 1)

    axes[row_index, 0].imshow(image, cmap="gray")
    axes[row_index, 0].set_title(
        f"{class_name.title()} - MRI",
        pad=10
    )

    axes[row_index, 1].imshow(mask, cmap="gray", vmin=0, vmax=1)
    axes[row_index, 1].set_title("Ground-truth mask", pad=10)

    axes[row_index, 2].imshow(image, cmap="gray")
    axes[row_index, 2].imshow(
        np.ma.masked_where(~mask, mask),
        cmap="autumn",
        alpha=0.4,
        vmin=0,
        vmax=1
    )
    axes[row_index, 2].contour(
        mask.astype(float),
        levels=[0.5],
        colors=["red"],
        linewidths=1.2
    )
    axes[row_index, 2].set_title("MRI + tumor overlay", pad=10)

    for ax in axes[row_index]:
        ax.axis("off")

fig.suptitle(
    "Brain Tumor MRI Dataset: Ground-Truth Segmentation",
    fontsize=17,
    fontweight="bold",
    y=0.98
)

fig.subplots_adjust(
    top=0.93,
    bottom=0.03,
    hspace=0.22,
    wspace=0.08
)

output_path = FIGURE_DIR / "mri_mask_visualization.png"

plt.savefig(output_path, dpi=300, bbox_inches="tight")
plt.show()

print(f"Visualization saved to: {output_path}")

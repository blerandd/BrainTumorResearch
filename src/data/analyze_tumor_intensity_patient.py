
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "processed"
OUTPUT_DIR = ROOT / "results" / "segmentation" / "diagnostics"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

metadata = pd.read_csv(DATA_DIR / "splits.csv")

records = []

for row in metadata.itertuples(index=False):
    image_path = DATA_DIR / row.image_path
    mask_path = DATA_DIR / row.mask_path

    image = np.asarray(Image.open(image_path))
    mask = np.asarray(Image.open(mask_path)) > 0

    if image.shape != mask.shape:
        raise ValueError(
            f"Image-mask dimension mismatch: {row.sample_id}"
        )

    image = image.astype(np.float64)

    tumor_pixels = image[mask]

    brain_region = image > 0
    surrounding_pixels = image[brain_region & ~mask]

    if tumor_pixels.size == 0 or surrounding_pixels.size == 0:
        raise ValueError(
            f"Invalid tumor or surrounding region: {row.sample_id}"
        )

    tumor_mean = float(tumor_pixels.mean())
    surrounding_mean = float(surrounding_pixels.mean())

    records.append({
        "sample_id": row.sample_id,
        "patient_id": row.patient_id,
        "label": row.label,
        "split": row.split,
        "tumor_mean": tumor_mean,
        "tumor_std": float(tumor_pixels.std()),
        "non_tumor_mean": surrounding_mean,
        "non_tumor_std": float(surrounding_pixels.std()),
        "intensity_difference": tumor_mean - surrounding_mean,
        "intensity_ratio": tumor_mean / max(surrounding_mean, 1.0)
    })

image_df = pd.DataFrame(records)

image_output = OUTPUT_DIR / "tumor_intensity_by_image.csv"
image_df.to_csv(image_output, index=False)

patient_df = image_df.groupby(
    ["label", "patient_id", "split"],
    as_index=False
).agg(
    number_of_images=("sample_id", "count"),
    tumor_intensity=("tumor_mean", "mean"),
    surrounding_intensity=("non_tumor_mean", "mean"),
    intensity_difference=("intensity_difference", "mean"),
    intensity_ratio=("intensity_ratio", "mean")
)

patient_output = OUTPUT_DIR / "patient_intensity_contrast.csv"
patient_df.to_csv(patient_output, index=False)

print("\nPATIENT-LEVEL INTENSITY ANALYSIS")
print("=" * 75)

for label in ["glioma", "meningioma", "pituitary"]:
    subset = patient_df[
        (patient_df["label"] == label) &
        (patient_df["split"] == "test")
    ]

    print(f"\n{label.upper()} — TEST PATIENTS")
    print("-" * 75)

    print(
        subset[
            [
                "patient_id",
                "number_of_images",
                "intensity_difference",
                "intensity_ratio"
            ]
        ].round(3).to_string(index=False)
    )

    print("\nCLASS STATISTICS")

    print(
        subset[
            ["intensity_difference", "intensity_ratio"]
        ].describe().round(3).to_string()
    )

print("\nSAVED FILES")
print(image_output)
print(patient_output)

print("\nINTENSITY ANALYSIS COMPLETED")

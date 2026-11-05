
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT / "data" / "processed"
RESULTS_DIR = (
    ROOT / "results" / "segmentation" / "statistical_analysis"
)

SPLITS_FILE = DATA_DIR / "splits.csv"
METRICS_FILE = (
    ROOT / "results" / "segmentation" /
    "patient_segmentation_metrics.csv"
)

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

FEATURES = [
    "tumor_area_percent",
    "perimeter_to_area",
    "compactness",
    "eccentricity",
    "component_count",
    "largest_component_fraction"
]


def calculate_mask_features(mask_path):
    mask = np.asarray(Image.open(mask_path)) > 0

    if mask.ndim > 2:
        mask = np.any(mask, axis=-1)

    area = int(mask.sum())

    if area == 0:
        return None

    structure = np.ones((3, 3), dtype=bool)
    eroded = ndimage.binary_erosion(
        mask,
        structure=structure,
        border_value=0
    )

    boundary = mask & ~eroded
    perimeter = int(boundary.sum())

    labels, component_count = ndimage.label(
        mask,
        structure=structure
    )

    component_sizes = np.bincount(labels.ravel())[1:]
    largest_component = (
        int(component_sizes.max()) if len(component_sizes) else 0
    )

    coordinates = np.argwhere(mask)

    if len(coordinates) > 1:
        covariance = np.cov(coordinates, rowvar=False)
        eigenvalues = np.linalg.eigvalsh(covariance)
        eigenvalues = np.maximum(eigenvalues, 0)

        if eigenvalues[-1] > 0:
            eccentricity = np.sqrt(
                1 - eigenvalues[0] / eigenvalues[-1]
            )
        else:
            eccentricity = 0.0
    else:
        eccentricity = 0.0

    compactness = (
        4 * np.pi * area / (perimeter ** 2)
        if perimeter > 0 else 0.0
    )

    return {
        "tumor_area_pixels": area,
        "tumor_area_percent": area / mask.size * 100,
        "perimeter_pixels": perimeter,
        "perimeter_to_area": perimeter / area,
        "compactness": compactness,
        "eccentricity": eccentricity,
        "component_count": int(component_count),
        "largest_component_fraction": largest_component / area
    }


splits = pd.read_csv(SPLITS_FILE)
patient_metrics = pd.read_csv(METRICS_FILE)

splits["patient_id"] = (
    splits["patient_id"].astype(str).str.strip()
)

patient_metrics["patient_id"] = (
    patient_metrics["patient_id"].astype(str).str.strip()
)

glioma_test = splits[
    (splits["label"].str.lower() == "glioma") &
    (splits["split"].str.lower() == "test")
].copy()

if glioma_test.empty:
    raise ValueError("No glioma test images were found.")

image_results = []

for row in glioma_test.itertuples():
    mask_path = DATA_DIR / row.mask_path

    if not mask_path.exists():
        raise FileNotFoundError(mask_path)

    features = calculate_mask_features(mask_path)

    if features is None:
        print(f"Skipping empty mask: {row.sample_id}")
        continue

    image_results.append({
        "sample_id": row.sample_id,
        "patient_id": row.patient_id,
        "label": row.label,
        **features
    })

image_df = pd.DataFrame(image_results)

image_output = RESULTS_DIR / "glioma_image_morphology.csv"
image_df.to_csv(image_output, index=False)

patient_morphology = (
    image_df.groupby("patient_id")[FEATURES]
    .median()
    .reset_index()
)

patient_counts = (
    image_df.groupby("patient_id")
    .size()
    .rename("number_of_images_analyzed")
    .reset_index()
)

patient_morphology = patient_morphology.merge(
    patient_counts,
    on="patient_id"
)

patient_morphology = patient_morphology.merge(
    patient_metrics[
        ["patient_id", "dice_mean", "iou_mean"]
    ],
    on="patient_id",
    how="inner",
    validate="one_to_one"
)

if len(patient_morphology) < 3:
    raise ValueError(
        "Too few matching patients for correlation analysis."
    )

patient_output = RESULTS_DIR / "glioma_patient_morphology.csv"
patient_morphology.to_csv(patient_output, index=False)

print("\nGLIOMA PATIENT-LEVEL MORPHOLOGY")
print("=" * 75)
print(patient_morphology.to_string(index=False))

print("\nSPEARMAN CORRELATION WITH PATIENT-LEVEL DICE")
print("=" * 75)

correlation_results = []

for feature in FEATURES:
    result = spearmanr(
        patient_morphology[feature],
        patient_morphology["dice_mean"]
    )

    correlation_results.append({
        "feature": feature,
        "spearman_rho": result.statistic,
        "p_value": result.pvalue,
        "patients": len(patient_morphology)
    })

correlations = pd.DataFrame(correlation_results)

sorted_indices = np.argsort(correlations["p_value"].values)
adjusted = np.empty(len(correlations))
running_max = 0.0
number_of_tests = len(correlations)

for rank, index in enumerate(sorted_indices):
    adjusted_value = (
        correlations.loc[index, "p_value"] *
        (number_of_tests - rank)
    )
    running_max = max(running_max, adjusted_value)
    adjusted[index] = min(running_max, 1.0)

correlations["holm_adjusted_p"] = adjusted

print(correlations.to_string(index=False))

correlations.to_csv(
    RESULTS_DIR / "glioma_morphology_correlations.csv",
    index=False
)

fig, axes = plt.subplots(2, 3, figsize=(16, 10))

for axis, feature in zip(axes.flat, FEATURES):
    axis.scatter(
        patient_morphology[feature],
        patient_morphology["dice_mean"],
        alpha=0.85
    )

    for row in patient_morphology.itertuples():
        axis.annotate(
            str(row.patient_id),
            (getattr(row, feature), row.dice_mean),
            fontsize=7
        )

    rho = correlations.loc[
        correlations["feature"] == feature,
        "spearman_rho"
    ].iloc[0]

    axis.set_title(f"{feature}\nSpearman rho = {rho:.3f}")
    axis.set_xlabel(feature)
    axis.set_ylabel("Patient-level Dice")
    axis.grid(alpha=0.3)

fig.suptitle(
    "Glioma Tumor Morphology vs U-Net Segmentation Performance",
    fontsize=15
)

plt.tight_layout()

plot_path = RESULTS_DIR / "glioma_morphology_vs_dice.png"

plt.savefig(
    plot_path,
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("\nSAVED RESULTS")
print(image_output)
print(patient_output)
print(RESULTS_DIR / "glioma_morphology_correlations.csv")
print(plot_path)

print("\nGLIOMA MORPHOLOGY ANALYSIS COMPLETED")

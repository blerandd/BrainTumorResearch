
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

from src.data.dataset_loader import BrainTumorDatasetLoader


BASE_DIR = Path(__file__).resolve().parents[2]

MODEL_PATH = (
    BASE_DIR
    / "models"
    / "segmentation"
    / "unet_best.keras"
)

MISSED_CASES_PATH = (
    BASE_DIR
    / "results"
    / "segmentation"
    / "boundary_metrics"
    / "infinite_distance_cases.csv"
)

OUTPUT_DIR = (
    BASE_DIR
    / "results"
    / "segmentation"
    / "missed_tumor_analysis"
)

MASK_THRESHOLD = 0.5


def load_test_data(loader):
    datasets = loader.get_segmentation_datasets()
    test_dataset = datasets["test"]

    images = []
    masks = []

    for batch_images, batch_masks in test_dataset:
        images.append(batch_images.numpy())
        masks.append(batch_masks.numpy())

    images = np.concatenate(images, axis=0)
    masks = np.concatenate(masks, axis=0)

    metadata = loader.metadata.copy()
    metadata = metadata[
        metadata["split"] == "test"
    ].reset_index(drop=True)

    if len(images) != len(masks) or len(images) != len(metadata):
        raise ValueError(
            f"Data mismatch: images={len(images)}, "
            f"masks={len(masks)}, metadata={len(metadata)}"
        )

    return images, masks, metadata


def load_missed_cases():
    if not MISSED_CASES_PATH.exists():
        raise FileNotFoundError(
            f"Missed cases file not found: {MISSED_CASES_PATH}"
        )

    missed_cases = pd.read_csv(MISSED_CASES_PATH)

    missed_cases["sample_id"] = (
        missed_cases["sample_id"].astype(str)
    )

    return missed_cases


def select_missed_images(images, masks, metadata, missed_cases):
    metadata = metadata.copy()
    metadata["sample_id"] = metadata["sample_id"].astype(str)

    missed_ids = set(missed_cases["sample_id"])

    indices = [
        index
        for index, sample_id in enumerate(metadata["sample_id"])
        if sample_id in missed_ids
    ]

    selected_metadata = metadata.iloc[indices].reset_index(drop=True)

    if len(indices) != len(missed_cases):
        found_ids = set(selected_metadata["sample_id"])
        missing_ids = missed_ids - found_ids

        raise ValueError(
            f"Could not match all missed cases. "
            f"Missing sample IDs: {sorted(missing_ids)}"
        )

    return (
        images[indices],
        masks[indices],
        selected_metadata
    )


def analyze_predictions(model, images, masks, metadata):
    predictions = model.predict(images, verbose=1)

    results = []

    for index in range(len(images)):
        true_mask = masks[index].squeeze() > MASK_THRESHOLD
        probability_map = predictions[index].squeeze()

        tumor_probabilities = probability_map[true_mask]

        if len(tumor_probabilities) == 0:
            raise ValueError(
                f"Empty ground-truth mask for sample "
                f"{metadata.iloc[index]['sample_id']}"
            )

        tumor_pixels = int(np.sum(true_mask))
        total_pixels = int(true_mask.size)

        row = metadata.iloc[index]

        results.append({
            "sample_id": row["sample_id"],
            "patient_id": row["patient_id"],
            "label": row["label"],

            "tumor_pixels": tumor_pixels,
            "tumor_area_percent": (
                tumor_pixels / total_pixels * 100
            ),

            "tumor_probability_mean": float(
                np.mean(tumor_probabilities)
            ),
            "tumor_probability_median": float(
                np.median(tumor_probabilities)
            ),
            "tumor_probability_max": float(
                np.max(tumor_probabilities)
            ),
            "tumor_probability_min": float(
                np.min(tumor_probabilities)
            ),

            "tumor_pixels_prob_ge_01": int(
                np.sum(tumor_probabilities >= 0.1)
            ),
            "tumor_pixels_prob_ge_025": int(
                np.sum(tumor_probabilities >= 0.25)
            ),
            "tumor_pixels_prob_ge_04": int(
                np.sum(tumor_probabilities >= 0.4)
            ),
            "tumor_pixels_prob_ge_05": int(
                np.sum(tumor_probabilities >= 0.5)
            ),

            "image_probability_mean": float(
                np.mean(probability_map)
            ),
            "image_probability_max": float(
                np.max(probability_map)
            ),
            "image_probability_std": float(
                np.std(probability_map)
            ),

            "predicted_pixels_at_05": int(
                np.sum(probability_map >= 0.5)
            )
        })

    return pd.DataFrame(results)


def create_class_summary(results):
    return (
        results
        .groupby("label")
        .agg(
            patient_count=("patient_id", "nunique"),
            image_count=("sample_id", "count"),
            mean_tumor_area_percent=("tumor_area_percent", "mean"),
            mean_tumor_probability=("tumor_probability_mean", "mean"),
            median_tumor_probability=("tumor_probability_median", "mean"),
            mean_max_tumor_probability=("tumor_probability_max", "mean"),
            mean_image_probability_max=("image_probability_max", "mean"),
            mean_image_probability=("image_probability_mean", "mean")
        )
        .reset_index()
    )


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading missed cases...")

    missed_cases = load_missed_cases()

    print(f"Missed cases found: {len(missed_cases)}")

    print("\nLoading U-Net model...")

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}"
        )

    model = tf.keras.models.load_model(
        MODEL_PATH,
        compile=False
    )

    print("\nLoading test dataset...")

    loader = BrainTumorDatasetLoader()

    images, masks, metadata = load_test_data(loader)

    print(f"Total test images: {len(images)}")

    print("\nSelecting completely missed tumors...")

    missed_images, missed_masks, missed_metadata = (
        select_missed_images(
            images,
            masks,
            metadata,
            missed_cases
        )
    )

    print(f"Selected images: {len(missed_images)}")

    print("\nAnalyzing prediction probabilities...")

    results = analyze_predictions(
        model,
        missed_images,
        missed_masks,
        missed_metadata
    )

    class_summary = create_class_summary(results)

    results.to_csv(
        OUTPUT_DIR / "missed_tumor_probability_analysis.csv",
        index=False
    )

    class_summary.to_csv(
        OUTPUT_DIR / "missed_tumor_analysis_by_class.csv",
        index=False
    )

    print("\nMISSED TUMOR ANALYSIS")
    print(results.to_string(index=False))

    print("\nSUMMARY BY TUMOR CLASS")
    print(class_summary.to_string(index=False))

    print("\nOVERALL STATISTICS")

    print(
        "Mean tumor area (%):",
        round(results["tumor_area_percent"].mean(), 4)
    )

    print(
        "Mean probability inside tumor:",
        round(results["tumor_probability_mean"].mean(), 6)
    )

    print(
        "Mean maximum probability inside tumor:",
        round(results["tumor_probability_max"].mean(), 6)
    )

    print(
        "Mean maximum probability across image:",
        round(results["image_probability_max"].mean(), 6)
    )

    print(
        "Tumor pixels with probability >= 0.4:",
        int(results["tumor_pixels_prob_ge_04"].sum())
    )

    print(
        "Tumor pixels with probability >= 0.5:",
        int(results["tumor_pixels_prob_ge_05"].sum())
    )

    print("\nAnalysis completed successfully.")
    print(f"Results saved in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

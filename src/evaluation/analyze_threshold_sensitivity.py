
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

from src.data.dataset_loader import BrainTumorDatasetLoader


BASE_DIR = Path(__file__).resolve().parents[2]
MODEL_PATH = BASE_DIR / "models" / "segmentation" / "unet_best.keras"
OUTPUT_DIR = (
    BASE_DIR
    / "results"
    / "segmentation"
    / "threshold_sensitivity"
)

THRESHOLDS = [0.1, 0.2, 0.3, 0.4, 0.5]
MASK_THRESHOLD = 0.5


def load_validation_data(loader):
    datasets = loader.get_segmentation_datasets()
    validation_dataset = datasets["validation"]

    images = []
    masks = []

    for batch_images, batch_masks in validation_dataset:
        images.append(batch_images.numpy())
        masks.append(batch_masks.numpy())

    images = np.concatenate(images, axis=0)
    masks = np.concatenate(masks, axis=0)

    metadata = loader.metadata.copy()
    metadata = (
        metadata[metadata["split"] == "validation"]
        .reset_index(drop=True)
    )

    if len(images) != len(masks) or len(images) != len(metadata):
        raise ValueError(
            f"Data mismatch: images={len(images)}, "
            f"masks={len(masks)}, metadata={len(metadata)}"
        )

    return images, masks, metadata


def calculate_metrics(true_mask, predicted_mask):
    true_mask = true_mask.astype(bool)
    predicted_mask = predicted_mask.astype(bool)

    true_positive = np.logical_and(
        true_mask, predicted_mask
    ).sum()

    false_positive = np.logical_and(
        ~true_mask, predicted_mask
    ).sum()

    false_negative = np.logical_and(
        true_mask, ~predicted_mask
    ).sum()

    denominator = (
        2 * true_positive + false_positive + false_negative
    )

    dice = (
        2 * true_positive / denominator
        if denominator > 0 else 1.0
    )

    precision = (
        true_positive / (true_positive + false_positive)
        if true_positive + false_positive > 0 else 0.0
    )

    recall = (
        true_positive / (true_positive + false_negative)
        if true_positive + false_negative > 0 else 0.0
    )

    return dice, precision, recall


def evaluate_threshold(probabilities, masks, metadata, threshold):
    results = []

    for index in range(len(probabilities)):
        true_mask = masks[index].squeeze() > MASK_THRESHOLD
        predicted_mask = probabilities[index].squeeze() >= threshold

        dice, precision, recall = calculate_metrics(
            true_mask,
            predicted_mask
        )

        row = metadata.iloc[index]

        results.append({
            "sample_id": row["sample_id"],
            "patient_id": row["patient_id"],
            "label": row["label"],
            "threshold": threshold,
            "dice": dice,
            "precision": precision,
            "recall": recall,
            "ground_truth_pixels": int(true_mask.sum()),
            "predicted_pixels": int(predicted_mask.sum()),
            "prediction_empty": not np.any(predicted_mask)
        })

    return pd.DataFrame(results)


def summarize_results(results):
    return pd.Series({
        "image_count": len(results),
        "dice_mean": results["dice"].mean(),
        "dice_std": results["dice"].std(),
        "dice_median": results["dice"].median(),
        "precision_mean": results["precision"].mean(),
        "recall_mean": results["recall"].mean(),
        "empty_predictions": int(
            results["prediction_empty"].sum()
        ),
        "empty_prediction_rate": (
            results["prediction_empty"].mean() * 100
        )
    })


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}"
        )

    print("Loading baseline U-Net...")
    model = tf.keras.models.load_model(
        MODEL_PATH,
        compile=False
    )

    print("Loading validation dataset...")
    loader = BrainTumorDatasetLoader()

    images, masks, metadata = load_validation_data(loader)

    print(f"Validation images: {len(images)}")
    print(f"Image shape: {images.shape}")
    print(f"Mask shape: {masks.shape}")

    print("\nGenerating probability predictions...")
    probabilities = model.predict(images, verbose=1)

    all_results = []
    overall_summaries = []
    class_summaries = []

    for threshold in THRESHOLDS:
        print(f"\nEvaluating threshold: {threshold}")

        threshold_results = evaluate_threshold(
            probabilities,
            masks,
            metadata,
            threshold
        )

        all_results.append(threshold_results)

        overall = summarize_results(threshold_results)
        overall["threshold"] = threshold
        overall_summaries.append(overall)

        grouped = (
            threshold_results
            .groupby("label")
            .apply(summarize_results, include_groups=False)
            .reset_index()
        )

        grouped["threshold"] = threshold
        class_summaries.append(grouped)

    image_results = pd.concat(
        all_results,
        ignore_index=True
    )

    overall_summary = pd.DataFrame(overall_summaries)

    class_summary = pd.concat(
        class_summaries,
        ignore_index=True
    )

    image_results.to_csv(
        OUTPUT_DIR / "threshold_results_by_image.csv",
        index=False
    )

    overall_summary.to_csv(
        OUTPUT_DIR / "threshold_results_overall.csv",
        index=False
    )

    class_summary.to_csv(
        OUTPUT_DIR / "threshold_results_by_class.csv",
        index=False
    )

    print("\nOVERALL THRESHOLD COMPARISON")
    print(
        overall_summary.to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}"
        )
    )

    print("\nRESULTS BY TUMOR CLASS")
    print(
        class_summary.to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}"
        )
    )

    print("\nThreshold sensitivity analysis completed.")
    print(f"Results saved in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

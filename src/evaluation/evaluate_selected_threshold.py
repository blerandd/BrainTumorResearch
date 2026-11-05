
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

from src.data.dataset_loader import BrainTumorDatasetLoader
from src.evaluation.boundary_metrics import calculate_all_boundary_metrics


BASE_DIR = Path(__file__).resolve().parents[2]

MODEL_PATH = (
    BASE_DIR
    / "models"
    / "segmentation"
    / "unet_best.keras"
)

OUTPUT_DIR = (
    BASE_DIR
    / "results"
    / "segmentation"
    / "final_evaluation"
)

SELECTED_THRESHOLD = 0.1
MASK_THRESHOLD = 0.5
BOUNDARY_TOLERANCE = 2.0


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
    metadata = (
        metadata[metadata["split"] == "test"]
        .reset_index(drop=True)
    )

    if not (
        len(images) == len(masks) == len(metadata)
    ):
        raise ValueError(
            f"Data mismatch: images={len(images)}, "
            f"masks={len(masks)}, metadata={len(metadata)}"
        )

    return images, masks, metadata


def calculate_segmentation_metrics(true_mask, predicted_mask):
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

    dice_denominator = (
        2 * true_positive + false_positive + false_negative
    )

    dice = (
        2 * true_positive / dice_denominator
        if dice_denominator > 0 else 1.0
    )

    union = true_positive + false_positive + false_negative

    iou = (
        true_positive / union
        if union > 0 else 1.0
    )

    precision = (
        true_positive / (true_positive + false_positive)
        if true_positive + false_positive > 0 else 0.0
    )

    recall = (
        true_positive / (true_positive + false_negative)
        if true_positive + false_negative > 0 else 0.0
    )

    return dice, iou, precision, recall


def evaluate_predictions(probabilities, masks, metadata):
    results = []

    for index in range(len(probabilities)):
        true_mask = (
            masks[index].squeeze() > MASK_THRESHOLD
        )

        predicted_mask = (
            probabilities[index].squeeze()
            >= SELECTED_THRESHOLD
        )

        dice, iou, precision, recall = (
            calculate_segmentation_metrics(
                true_mask,
                predicted_mask
            )
        )

        boundary_metrics = calculate_all_boundary_metrics(
            true_mask,
            predicted_mask,
            tolerance=BOUNDARY_TOLERANCE
        )

        row = metadata.iloc[index]

        results.append({
            "sample_id": row["sample_id"],
            "patient_id": row["patient_id"],
            "label": row["label"],
            "threshold": SELECTED_THRESHOLD,
            "dice": dice,
            "iou": iou,
            "precision": precision,
            "recall": recall,
            "hd95_pixels": boundary_metrics["hd95_pixels"],
            "assd_pixels": boundary_metrics["assd_pixels"],
            "boundary_dice": boundary_metrics["boundary_dice"],
            "ground_truth_pixels": int(true_mask.sum()),
            "predicted_pixels": int(predicted_mask.sum()),
            "ground_truth_empty": not np.any(true_mask),
            "prediction_empty": not np.any(predicted_mask)
        })

    return pd.DataFrame(results)


def summarize_group(group):
    hd95 = group["hd95_pixels"].to_numpy(dtype=float)
    assd = group["assd_pixels"].to_numpy(dtype=float)

    finite_hd95 = hd95[np.isfinite(hd95)]
    finite_assd = assd[np.isfinite(assd)]

    return pd.Series({
        "image_count": len(group),
        "dice_mean": group["dice"].mean(),
        "dice_std": group["dice"].std(),
        "dice_median": group["dice"].median(),
        "iou_mean": group["iou"].mean(),
        "precision_mean": group["precision"].mean(),
        "recall_mean": group["recall"].mean(),
        "hd95_mean_pixels": (
            np.mean(finite_hd95)
            if len(finite_hd95) else np.nan
        ),
        "hd95_median_pixels": (
            np.median(finite_hd95)
            if len(finite_hd95) else np.nan
        ),
        "hd95_infinite_count": int(np.isinf(hd95).sum()),
        "assd_mean_pixels": (
            np.mean(finite_assd)
            if len(finite_assd) else np.nan
        ),
        "assd_median_pixels": (
            np.median(finite_assd)
            if len(finite_assd) else np.nan
        ),
        "assd_infinite_count": int(np.isinf(assd).sum()),
        "boundary_dice_mean": group["boundary_dice"].mean(),
        "boundary_dice_std": group["boundary_dice"].std(),
        "empty_predictions": int(
            group["prediction_empty"].sum()
        ),
        "empty_prediction_rate": (
            group["prediction_empty"].mean() * 100
        )
    })


def create_patient_summary(results):
    return (
        results
        .groupby(["patient_id", "label"])
        .apply(summarize_group, include_groups=False)
        .reset_index()
    )


def create_class_summary(results):
    return (
        results
        .groupby("label")
        .apply(summarize_group, include_groups=False)
        .reset_index()
    )


def create_overall_summary(results):
    summary = summarize_group(results).to_frame().T
    summary.insert(0, "evaluation", "test_set")
    summary.insert(1, "threshold", SELECTED_THRESHOLD)
    return summary


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

    print("Loading untouched test dataset...")
    loader = BrainTumorDatasetLoader()

    images, masks, metadata = load_test_data(loader)

    print(f"Test images: {len(images)}")
    print(f"Image shape: {images.shape}")
    print(f"Mask shape: {masks.shape}")
    print(f"Selected threshold: {SELECTED_THRESHOLD}")

    print("\nGenerating test predictions...")
    probabilities = model.predict(images, verbose=1)

    print("\nCalculating final segmentation metrics...")
    results = evaluate_predictions(
        probabilities,
        masks,
        metadata
    )

    patient_summary = create_patient_summary(results)
    class_summary = create_class_summary(results)
    overall_summary = create_overall_summary(results)

    results.to_csv(
        OUTPUT_DIR / "final_metrics_by_image.csv",
        index=False
    )

    patient_summary.to_csv(
        OUTPUT_DIR / "final_metrics_by_patient.csv",
        index=False
    )

    class_summary.to_csv(
        OUTPUT_DIR / "final_metrics_by_class.csv",
        index=False
    )

    overall_summary.to_csv(
        OUTPUT_DIR / "final_metrics_overall.csv",
        index=False
    )

    print("\nFINAL OVERALL TEST RESULTS")
    print(
        overall_summary.to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}"
        )
    )

    print("\nFINAL RESULTS BY TUMOR CLASS")
    print(
        class_summary.to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}"
        )
    )

    print("\nRESULTS BY PATIENT")
    print(
        patient_summary.to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}"
        )
    )

    print("\nEMPTY PREDICTIONS")
    print(
        int(results["prediction_empty"].sum()),
        "out of",
        len(results)
    )

    print("\nINFINITE DISTANCE CASES")
    print(
        "HD95:",
        int(np.isinf(results["hd95_pixels"]).sum())
    )
    print(
        "ASSD:",
        int(np.isinf(results["assd_pixels"]).sum())
    )

    print("\nFinal test evaluation completed.")
    print(f"Results saved in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()


from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

from src.data.dataset_loader import BrainTumorDatasetLoader
from src.evaluation.boundary_metrics import calculate_all_boundary_metrics


BASE_DIR = Path(__file__).resolve().parents[2]
MODEL_PATH = BASE_DIR / "models" / "segmentation" / "unet_best.keras"
OUTPUT_DIR = BASE_DIR / "results" / "segmentation" / "boundary_metrics"

BOUNDARY_TOLERANCE = 2.0
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
    metadata = metadata[metadata["split"] == "test"].reset_index(drop=True)

    if len(images) != len(masks) or len(images) != len(metadata):
        raise ValueError(
            f"Data mismatch: images={len(images)}, "
            f"masks={len(masks)}, metadata={len(metadata)}"
        )

    return images, masks, metadata


def evaluate_boundary_metrics(model, images, masks, metadata):
    predictions = model.predict(images, verbose=1)

    results = []

    for index in range(len(images)):
        true_mask = masks[index].squeeze()
        predicted_mask = predictions[index].squeeze()

        true_mask = true_mask > MASK_THRESHOLD
        predicted_mask = predicted_mask > MASK_THRESHOLD

        metrics = calculate_all_boundary_metrics(
            true_mask,
            predicted_mask,
            tolerance=BOUNDARY_TOLERANCE
        )

        row = metadata.iloc[index]

        results.append({
            "sample_id": row["sample_id"],
            "patient_id": row["patient_id"],
            "label": row["label"],
            "hd95_pixels": metrics["hd95_pixels"],
            "assd_pixels": metrics["assd_pixels"],
            "boundary_dice": metrics["boundary_dice"],
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
        "hd95_mean_pixels": (
            np.mean(finite_hd95) if len(finite_hd95) else np.nan
        ),
        "hd95_median_pixels": (
            np.median(finite_hd95) if len(finite_hd95) else np.nan
        ),
        "hd95_infinite_count": int(np.isinf(hd95).sum()),
        "assd_mean_pixels": (
            np.mean(finite_assd) if len(finite_assd) else np.nan
        ),
        "assd_median_pixels": (
            np.median(finite_assd) if len(finite_assd) else np.nan
        ),
        "assd_infinite_count": int(np.isinf(assd).sum()),
        "boundary_dice_mean": group["boundary_dice"].mean(),
        "boundary_dice_std": group["boundary_dice"].std()
    })


def create_patient_summary(image_results):
    return (
        image_results
        .groupby(["patient_id", "label"])
        .apply(summarize_group, include_groups=False)
        .reset_index()
    )


def create_class_summary(image_results):
    return (
        image_results
        .groupby("label")
        .apply(summarize_group, include_groups=False)
        .reset_index()
    )


def create_overall_summary(image_results):
    summary = summarize_group(image_results).to_frame().T
    summary.insert(0, "evaluation", "test_set")
    return summary


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading U-Net model...")

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    model = tf.keras.models.load_model(
        MODEL_PATH,
        compile=False
    )

    print("Loading test dataset...")

    loader = BrainTumorDatasetLoader()
    images, masks, metadata = load_test_data(loader)

    print(f"Test images: {len(images)}")
    print(f"Image shape: {images.shape}")
    print(f"Mask shape: {masks.shape}")

    print("\nCalculating boundary metrics...")

    image_results = evaluate_boundary_metrics(
        model,
        images,
        masks,
        metadata
    )

    patient_summary = create_patient_summary(image_results)
    class_summary = create_class_summary(image_results)
    overall_summary = create_overall_summary(image_results)

    image_results.to_csv(
        OUTPUT_DIR / "boundary_metrics_by_image.csv",
        index=False
    )

    patient_summary.to_csv(
        OUTPUT_DIR / "boundary_metrics_by_patient.csv",
        index=False
    )

    class_summary.to_csv(
        OUTPUT_DIR / "boundary_metrics_by_class.csv",
        index=False
    )

    overall_summary.to_csv(
        OUTPUT_DIR / "boundary_metrics_overall.csv",
        index=False
    )

    print("\nOVERALL TEST RESULTS")
    print(overall_summary.to_string(index=False))

    print("\nRESULTS BY TUMOR CLASS")
    print(class_summary.to_string(index=False))

    print("\nRESULTS BY PATIENT")
    print(patient_summary.to_string(index=False))

    print("\nINFINITE DISTANCE CASES")
    print(
        "HD95:",
        int(np.isinf(image_results["hd95_pixels"]).sum())
    )
    print(
        "ASSD:",
        int(np.isinf(image_results["assd_pixels"]).sum())
    )

    print("\nAll boundary metric evaluations completed.")
    print(f"Results saved in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

from pathlib import Path
import gc

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

from src.data.dataset_loader import BrainTumorDatasetLoader
from src.models.unet import (
    dice_coefficient,
    dice_loss,
    combined_segmentation_loss
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

BASELINE_MODEL_PATH = (
    PROJECT_ROOT / "models" / "segmentation" / "unet_best.keras"
)

AUGMENTED_MODEL_PATH = (
    PROJECT_ROOT / "models" / "segmentation" / "unet_augmented_best.keras"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "segmentation"
    / "statistical_analysis"
    / "glioma_patient_failures"
    / "focused_patient_investigation"
)

PATIENTS = [
    "MR048621B",
    "101020",
    "MR049104"
]

THRESHOLD = 0.5
BATCH_SIZE = 8


def calculate_dice(y_true, y_pred):
    y_true = y_true.astype(bool)
    y_pred = y_pred.astype(bool)

    denominator = np.sum(y_true) + np.sum(y_pred)

    if denominator == 0:
        return 1.0

    intersection = np.sum(y_true & y_pred)

    return float(2.0 * intersection / denominator)


def load_test_data():
    print("\nLOADING TEST DATA")
    print("=" * 70)

    loader = BrainTumorDatasetLoader(
        processed_dir="data/processed",
        image_size=(256, 256),
        batch_size=16,
        seed=42
    )

    test_dataset = loader.get_segmentation_datasets()["test"]

    images = []
    masks = []

    for batch_images, batch_masks in test_dataset:
        images.append(batch_images.numpy())
        masks.append(batch_masks.numpy())

    images = np.concatenate(images, axis=0)
    masks = np.concatenate(masks, axis=0)

    metadata = (
        loader.metadata[
            loader.metadata["split"] == "test"
        ]
        .reset_index(drop=True)
        .copy()
    )

    if len(images) != len(metadata):
        raise ValueError(
            "Test image count does not match metadata count."
        )

    metadata["patient_id"] = metadata["patient_id"].astype(str)

    print(f"Test images: {len(images)}")
    print(f"Test patients: {metadata['patient_id'].nunique()}")

    return images, masks, metadata


def load_model(path, name):
    if not path.is_file():
        raise FileNotFoundError(
            f"{name} model not found: {path}"
        )

    print(f"\nLoading {name} model...")

    return tf.keras.models.load_model(
        path,
        custom_objects={
            "dice_coefficient": dice_coefficient,
            "dice_loss": dice_loss,
            "combined_segmentation_loss": combined_segmentation_loss
        },
        compile=False
    )


def create_visualization(
    image,
    ground_truth,
    baseline_probability,
    augmented_probability,
    baseline_prediction,
    augmented_prediction,
    patient_id,
    sample_id,
    baseline_dice,
    augmented_dice,
    output_path
):
    figure, axes = plt.subplots(1, 6, figsize=(19, 4))

    panels = [
        (image.squeeze(), "Original MRI", "gray"),
        (ground_truth, "Ground Truth", "gray"),
        (baseline_probability, "Baseline Probability", "viridis"),
        (baseline_prediction, "Baseline Prediction", "gray"),
        (augmented_probability, "Augmented Probability", "viridis"),
        (augmented_prediction, "Augmented Prediction", "gray")
    ]

    for axis, (data, title, cmap) in zip(axes, panels):
        axis.imshow(data, cmap=cmap)
        axis.set_title(title, fontsize=9)
        axis.axis("off")

    figure.suptitle(
        f"Patient {patient_id} | Sample {sample_id} | "
        f"Baseline Dice: {baseline_dice:.4f} | "
        f"Augmented Dice: {augmented_dice:.4f} | "
        f"Change: {augmented_dice - baseline_dice:+.4f}",
        fontsize=11
    )

    figure.tight_layout()

    figure.savefig(
        output_path,
        dpi=180,
        bbox_inches="tight"
    )

    plt.close(figure)


def analyze_patient(
    patient_id,
    images,
    masks,
    metadata,
    baseline_model,
    augmented_model
):
    patient_dir = OUTPUT_DIR / patient_id
    patient_dir.mkdir(parents=True, exist_ok=True)

    patient_indices = np.where(
        metadata["patient_id"].to_numpy() == patient_id
    )[0]

    if len(patient_indices) == 0:
        print(f"\nWARNING: Patient {patient_id} not found.")
        return None

    patient_images = images[patient_indices]
    patient_masks = masks[patient_indices]

    patient_metadata = (
        metadata.iloc[patient_indices]
        .reset_index(drop=True)
    )

    print(f"\nPATIENT {patient_id}")
    print("=" * 70)
    print(f"Class: {patient_metadata['label'].iloc[0]}")
    print(f"Number of slices: {len(patient_indices)}")

    print("\nGenerating baseline predictions...")

    baseline_probabilities = baseline_model.predict(
        patient_images,
        batch_size=BATCH_SIZE,
        verbose=0
    )

    print("Generating augmented predictions...")

    augmented_probabilities = augmented_model.predict(
        patient_images,
        batch_size=BATCH_SIZE,
        verbose=0
    )

    records = []

    for index in range(len(patient_images)):
        image = patient_images[index].squeeze()

        ground_truth = (
            patient_masks[index].squeeze() >= THRESHOLD
        )

        baseline_probability = (
            baseline_probabilities[index].squeeze()
        )

        augmented_probability = (
            augmented_probabilities[index].squeeze()
        )

        baseline_prediction = (
            baseline_probability >= THRESHOLD
        )

        augmented_prediction = (
            augmented_probability >= THRESHOLD
        )

        baseline_dice = calculate_dice(
            ground_truth,
            baseline_prediction
        )

        augmented_dice = calculate_dice(
            ground_truth,
            augmented_prediction
        )

        tumor_probability_baseline = (
            float(np.mean(baseline_probability[ground_truth]))
            if np.any(ground_truth)
            else np.nan
        )

        tumor_probability_augmented = (
            float(np.mean(augmented_probability[ground_truth]))
            if np.any(ground_truth)
            else np.nan
        )

        record = {
            "patient_id": patient_id,
            "sample_id": patient_metadata.iloc[index]["sample_id"],
            "label": patient_metadata.iloc[index]["label"],
            "baseline_dice": baseline_dice,
            "augmented_dice": augmented_dice,
            "dice_change": augmented_dice - baseline_dice,
            "ground_truth_pixels": int(np.sum(ground_truth)),
            "baseline_predicted_pixels": int(
                np.sum(baseline_prediction)
            ),
            "augmented_predicted_pixels": int(
                np.sum(augmented_prediction)
            ),
            "baseline_empty_prediction": bool(
                not np.any(baseline_prediction)
            ),
            "augmented_empty_prediction": bool(
                not np.any(augmented_prediction)
            ),
            "baseline_tumor_mean_probability": tumor_probability_baseline,
            "augmented_tumor_mean_probability": tumor_probability_augmented,
            "baseline_tumor_max_probability": float(
                np.max(baseline_probability[ground_truth])
            ) if np.any(ground_truth) else np.nan,
            "augmented_tumor_max_probability": float(
                np.max(augmented_probability[ground_truth])
            ) if np.any(ground_truth) else np.nan
        }

        records.append(record)

        sample_id = patient_metadata.iloc[index]["sample_id"]

        output_path = (
            patient_dir / f"sample_{sample_id}_comparison.png"
        )

        create_visualization(
            image=image,
            ground_truth=ground_truth,
            baseline_probability=baseline_probability,
            augmented_probability=augmented_probability,
            baseline_prediction=baseline_prediction,
            augmented_prediction=augmented_prediction,
            patient_id=patient_id,
            sample_id=sample_id,
            baseline_dice=baseline_dice,
            augmented_dice=augmented_dice,
            output_path=output_path
        )

    del baseline_probabilities, augmented_probabilities
    gc.collect()

    return pd.DataFrame(records)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("\nFOCUSED GLIOMA PATIENT FAILURE INVESTIGATION")
    print("=" * 75)

    images, masks, metadata = load_test_data()

    baseline_model = load_model(
        BASELINE_MODEL_PATH,
        "Baseline"
    )

    augmented_model = load_model(
        AUGMENTED_MODEL_PATH,
        "Augmented"
    )

    all_results = []

    for patient_id in PATIENTS:
        patient_results = analyze_patient(
            patient_id,
            images,
            masks,
            metadata,
            baseline_model,
            augmented_model
        )

        if patient_results is not None:
            all_results.append(patient_results)

    if not all_results:
        raise ValueError("No selected patients were found.")

    results = pd.concat(
        all_results,
        ignore_index=True
    )

    patient_summary = (
        results.groupby(
            ["patient_id", "label"],
            as_index=False
        )
        .agg(
            image_count=("sample_id", "count"),
            baseline_dice_mean=("baseline_dice", "mean"),
            augmented_dice_mean=("augmented_dice", "mean"),
            mean_dice_change=("dice_change", "mean"),
            baseline_empty_predictions=(
                "baseline_empty_prediction", "sum"
            ),
            augmented_empty_predictions=(
                "augmented_empty_prediction", "sum"
            ),
            baseline_tumor_probability_mean=(
                "baseline_tumor_mean_probability", "mean"
            ),
            augmented_tumor_probability_mean=(
                "augmented_tumor_mean_probability", "mean"
            )
        )
    )

    results.to_csv(
        OUTPUT_DIR / "focused_patient_image_results.csv",
        index=False
    )

    patient_summary.to_csv(
        OUTPUT_DIR / "focused_patient_summary.csv",
        index=False
    )

    print("\nPATIENT SUMMARY")
    print("=" * 100)
    print(patient_summary.to_string(index=False))

    print("\nIMAGE-LEVEL RESULTS")
    print("=" * 100)
    print(results.to_string(index=False))

    print("\nSAVED RESULTS")
    print("=" * 75)
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Image-level CSV: {OUTPUT_DIR / 'focused_patient_image_results.csv'}")
    print(f"Patient summary: {OUTPUT_DIR / 'focused_patient_summary.csv'}")

    print("\nFOCUSED PATIENT INVESTIGATION COMPLETED")

    del baseline_model, augmented_model
    gc.collect()


if __name__ == "__main__":
    main()
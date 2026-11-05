
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from src.data.dataset_loader import BrainTumorDatasetLoader


PROJECT_ROOT = Path(__file__).resolve().parents[2]

BASELINE_MODEL_PATH = (
    PROJECT_ROOT / "models/segmentation/unet_best.keras"
)

AUGMENTED_MODEL_PATH = (
    PROJECT_ROOT / "models/segmentation/unet_augmented_best.keras"
)

PAIRED_RESULTS_PATH = (
    PROJECT_ROOT
    / "results/segmentation/baseline_vs_augmented/"
    / "paired_image_comparison.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results/segmentation/statistical_analysis/"
    / "augmentation_morphology/patient_visualizations"
)

THRESHOLD = 0.5

SELECTED_PATIENTS = {
    "92897": "improvement",
    "MR031409B": "improvement",
    "MR048621B": "decline",
    "101020": "decline"
}


def calculate_dice(y_true, y_pred):
    y_true = y_true.astype(np.float32).ravel()
    y_pred = y_pred.astype(np.float32).ravel()

    intersection = np.sum(y_true * y_pred)

    return (
        2.0 * intersection + 1e-7
    ) / (
        np.sum(y_true) + np.sum(y_pred) + 1e-7
    )


def load_test_data():
    loader = BrainTumorDatasetLoader(
        processed_dir="data/processed",
        image_size=(256, 256),
        batch_size=4,
        seed=42
    )

    dataset = loader.get_segmentation_datasets()["test"]

    images = []
    masks = []

    for batch_images, batch_masks in dataset:
        images.append(batch_images.numpy())
        masks.append(batch_masks.numpy())

    images = np.concatenate(images, axis=0)
    masks = np.concatenate(masks, axis=0)

    test_records = (
        loader.metadata[
            loader.metadata["split"] == "test"
        ]
        .reset_index(drop=True)
    )

    if len(images) != len(test_records):
        raise ValueError(
            "Test images and metadata counts do not match."
        )

    return images, masks, test_records


def select_patient_images(paired_results):
    selected_rows = []

    paired_results["patient_id"] = (
        paired_results["patient_id"].astype(str)
    )

    paired_results["sample_id"] = (
        paired_results["sample_id"].astype(str)
    )

    for patient_id, selection_type in SELECTED_PATIENTS.items():
        patient_rows = paired_results[
            paired_results["patient_id"] == patient_id
        ]

        if patient_rows.empty:
            raise ValueError(
                f"Patient {patient_id} was not found "
                "in paired_image_comparison.csv."
            )

        if selection_type == "improvement":
            selected = patient_rows.loc[
                patient_rows["delta_dice"].idxmax()
            ]
        else:
            selected = patient_rows.loc[
                patient_rows["delta_dice"].idxmin()
            ]

        selected_rows.append(selected)

    return pd.DataFrame(selected_rows).reset_index(drop=True)


def create_comparison_visualization(
    image,
    ground_truth,
    baseline_prediction,
    augmented_prediction,
    patient_id,
    sample_id,
    label,
    baseline_dice,
    augmented_dice,
    delta_dice
):
    figure, axes = plt.subplots(
        1,
        5,
        figsize=(19, 5)
    )

    image = image.squeeze()
    ground_truth = ground_truth.squeeze()
    baseline_prediction = baseline_prediction.squeeze()
    augmented_prediction = augmented_prediction.squeeze()

    axes[0].imshow(image, cmap="gray")
    axes[0].set_title("Original MRI")

    axes[1].imshow(ground_truth, cmap="gray")
    axes[1].set_title("Ground Truth")

    axes[2].imshow(baseline_prediction, cmap="gray")
    axes[2].set_title(
        f"Baseline U-Net\nDice: {baseline_dice:.4f}"
    )

    axes[3].imshow(augmented_prediction, cmap="gray")
    axes[3].set_title(
        f"Augmented U-Net\nDice: {augmented_dice:.4f}"
    )

    axes[4].imshow(image, cmap="gray")
    axes[4].imshow(
        ground_truth,
        cmap="Greens",
        alpha=0.35
    )
    axes[4].imshow(
        baseline_prediction,
        cmap="Reds",
        alpha=0.35
    )
    axes[4].imshow(
        augmented_prediction,
        cmap="Blues",
        alpha=0.35
    )
    axes[4].set_title(
        "Overlay\nGreen: GT | Red: Baseline | Blue: Augmented"
    )

    for axis in axes:
        axis.axis("off")

    figure.suptitle(
        f"Patient {patient_id} | Sample {sample_id} | "
        f"Class: {label} | Dice change: {delta_dice:+.4f}",
        fontsize=13
    )

    figure.tight_layout()

    filename = (
        f"patient_{patient_id}_sample_{sample_id}.png"
    )

    figure.savefig(
        OUTPUT_DIR / filename,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(figure)

    return filename


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("\nAUGMENTATION PATIENT VISUAL COMPARISON")
    print("=" * 70)

    for path in [
        BASELINE_MODEL_PATH,
        AUGMENTED_MODEL_PATH,
        PAIRED_RESULTS_PATH
    ]:
        if not path.is_file():
            raise FileNotFoundError(
                f"Required file not found: {path}"
            )

    print("\nLOADING TEST DATA")
    print("=" * 70)

    images, masks, test_records = load_test_data()

    print(f"Test images: {len(images)}")
    print(f"Test patients: {test_records['patient_id'].nunique()}")

    print("\nLOADING PAIRED RESULTS")
    print("=" * 70)

    paired_results = pd.read_csv(PAIRED_RESULTS_PATH)

    selected = select_patient_images(paired_results)

    test_records["sample_id"] = (
        test_records["sample_id"].astype(str)
    )

    selected["sample_id"] = selected["sample_id"].astype(str)

    indices = []

    for _, row in selected.iterrows():
        matches = test_records.index[
            test_records["sample_id"] == row["sample_id"]
        ].tolist()

        if len(matches) != 1:
            raise ValueError(
                f"Expected exactly one test image for "
                f"sample {row['sample_id']}, found {len(matches)}."
            )

        indices.append(matches[0])

    selected_images = images[indices]
    selected_masks = masks[indices]

    print("\nLOADING BASELINE MODEL")
    print("=" * 70)

    baseline_model = tf.keras.models.load_model(
        BASELINE_MODEL_PATH,
        compile=False
    )

    print("\nLOADING AUGMENTED MODEL")
    print("=" * 70)

    augmented_model = tf.keras.models.load_model(
        AUGMENTED_MODEL_PATH,
        compile=False
    )

    print("\nGENERATING BASELINE PREDICTIONS")
    print("=" * 70)

    baseline_probabilities = baseline_model.predict(
        selected_images,
        batch_size=4,
        verbose=1
    )

    print("\nGENERATING AUGMENTED PREDICTIONS")
    print("=" * 70)

    augmented_probabilities = augmented_model.predict(
        selected_images,
        batch_size=4,
        verbose=1
    )

    baseline_predictions = (
        baseline_probabilities >= THRESHOLD
    ).astype(np.uint8)

    augmented_predictions = (
        augmented_probabilities >= THRESHOLD
    ).astype(np.uint8)

    ground_truth = (
        selected_masks >= THRESHOLD
    ).astype(np.uint8)

    results = []

    print("\nCREATING PATIENT VISUALIZATIONS")
    print("=" * 70)

    for index, row in selected.iterrows():
        baseline_dice = calculate_dice(
            ground_truth[index],
            baseline_predictions[index]
        )

        augmented_dice = calculate_dice(
            ground_truth[index],
            augmented_predictions[index]
        )

        delta_dice = augmented_dice - baseline_dice

        record = test_records.iloc[indices[index]]

        filename = create_comparison_visualization(
            image=selected_images[index],
            ground_truth=ground_truth[index],
            baseline_prediction=baseline_predictions[index],
            augmented_prediction=augmented_predictions[index],
            patient_id=str(row["patient_id"]),
            sample_id=str(row["sample_id"]),
            label=record["label"],
            baseline_dice=baseline_dice,
            augmented_dice=augmented_dice,
            delta_dice=delta_dice
        )

        results.append({
            "patient_id": row["patient_id"],
            "sample_id": row["sample_id"],
            "label": record["label"],
            "baseline_dice": baseline_dice,
            "augmented_dice": augmented_dice,
            "dice_improvement": delta_dice,
            "visualization": filename
        })

        print(
            f"Patient {row['patient_id']}: "
            f"Baseline={baseline_dice:.4f}, "
            f"Augmented={augmented_dice:.4f}, "
            f"Change={delta_dice:+.4f}"
        )

    results_df = pd.DataFrame(results)

    results_df.to_csv(
        OUTPUT_DIR / "selected_patient_visualization_metrics.csv",
        index=False
    )

    print("\nSAVED VISUALIZATIONS")
    print("=" * 70)
    print(f"Directory: {OUTPUT_DIR}")
    print(
        f"Metrics: "
        f"{OUTPUT_DIR / 'selected_patient_visualization_metrics.csv'}"
    )

    print("\nPATIENT VISUAL COMPARISON COMPLETED SUCCESSFULLY")


if __name__ == "__main__":
    main()

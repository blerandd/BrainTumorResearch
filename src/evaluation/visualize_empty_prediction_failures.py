
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from src.data.dataset_loader import BrainTumorDatasetLoader


BASE_DIR = Path(__file__).resolve().parents[2]

MODEL_PATH = BASE_DIR / "models" / "segmentation" / "unet_best.keras"

OUTPUT_DIR = (
    BASE_DIR
    / "results"
    / "segmentation"
    / "diagnostics"
    / "empty_prediction_confidence"
    / "visualizations"
)

CONFIDENCE_FILE = (
    BASE_DIR
    / "results"
    / "segmentation"
    / "diagnostics"
    / "empty_prediction_confidence"
    / "empty_prediction_cases.csv"
)

MASK_THRESHOLD = 0.5


def load_test_data():
    loader = BrainTumorDatasetLoader()
    datasets = loader.get_segmentation_datasets()

    images = []
    masks = []

    for batch_images, batch_masks in datasets["test"]:
        images.append(batch_images.numpy())
        masks.append(batch_masks.numpy())

    images = np.concatenate(images, axis=0)
    masks = np.concatenate(masks, axis=0)

    metadata = loader.metadata.copy()
    metadata = metadata[
        metadata["split"] == "test"
    ].reset_index(drop=True)

    if len(images) != len(metadata):
        raise ValueError(
            f"Image and metadata mismatch: "
            f"{len(images)} images, {len(metadata)} metadata rows"
        )

    return images, masks, metadata


def create_failure_visualization(
    image,
    true_mask,
    probability_map,
    row,
    output_path
):
    image = image.squeeze()
    true_mask = true_mask.squeeze()
    probability_map = probability_map.squeeze()

    predicted_mask = probability_map > MASK_THRESHOLD

    maximum_probability = float(np.max(probability_map))
    tumor_probabilities = probability_map[true_mask > 0]

    tumor_mean_probability = (
        float(np.mean(tumor_probabilities))
        if len(tumor_probabilities) > 0
        else 0.0
    )

    tumor_max_probability = (
        float(np.max(tumor_probabilities))
        if len(tumor_probabilities) > 0
        else 0.0
    )

    figure, axes = plt.subplots(1, 5, figsize=(22, 5))

    axes[0].imshow(image, cmap="gray")
    axes[0].set_title("Original MRI")

    axes[1].imshow(true_mask, cmap="gray", vmin=0, vmax=1)
    axes[1].set_title("Ground Truth")

    probability_display = axes[2].imshow(
        probability_map,
        cmap="inferno",
        vmin=0,
        vmax=0.1
    )
    axes[2].set_title(
        f"Probability Map\nMax: {maximum_probability:.5f}"
    )
    figure.colorbar(
        probability_display,
        ax=axes[2],
        fraction=0.046,
        pad=0.04
    )

    axes[3].imshow(predicted_mask, cmap="gray", vmin=0, vmax=1)
    axes[3].set_title("Binary Prediction")

    axes[4].imshow(image, cmap="gray")

    axes[4].imshow(
        probability_map,
        cmap="inferno",
        alpha=0.45,
        vmin=0,
        vmax=1
    )

    axes[4].contour(
        true_mask,
        levels=[0.5],
        colors=["cyan"],
        linewidths=1
    )

    axes[4].set_title("MRI + Probability + GT Boundary")

    for axis in axes:
        axis.axis("off")

    figure.suptitle(
        f"Patient: {row['patient_id']} | "
        f"Sample: {row['sample_id']} | "
        f"Class: {row['label']}\n"
        f"Mean tumor probability: {tumor_mean_probability:.6f} | "
        f"Max tumor probability: {tumor_max_probability:.6f}",
        fontsize=13
    )

    figure.tight_layout()

    figure.savefig(
        output_path,
        dpi=180,
        bbox_inches="tight"
    )

    plt.close(figure)

    return {
        "sample_id": row["sample_id"],
        "patient_id": row["patient_id"],
        "label": row["label"],
        "max_probability": maximum_probability,
        "tumor_mean_probability": tumor_mean_probability,
        "tumor_max_probability": tumor_max_probability,
        "output_file": str(output_path)
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    if not CONFIDENCE_FILE.exists():
        raise FileNotFoundError(
            f"Confidence results not found: {CONFIDENCE_FILE}"
        )

    print("Loading baseline U-Net...")

    model = tf.keras.models.load_model(
        MODEL_PATH,
        compile=False
    )

    print("Loading test dataset...")

    images, masks, metadata = load_test_data()

    confidence_data = pd.read_csv(CONFIDENCE_FILE)

    failure_ids = set(
        confidence_data["sample_id"].tolist()
    )

    print(f"Test images: {len(images)}")
    print(f"Recorded empty predictions: {len(failure_ids)}")

    print("\nGenerating failure visualizations...")

    predictions = model.predict(images, verbose=1)

    results = []

    for index in range(len(images)):
        row = metadata.iloc[index]

        if row["sample_id"] not in failure_ids:
            continue

        probability_map = predictions[index].squeeze()
        predicted_mask = probability_map > MASK_THRESHOLD

        if np.any(predicted_mask):
            print(
                f"Warning: sample {row['sample_id']} "
                "is no longer an empty prediction."
            )
            continue

        filename = (
            f"patient_{row['patient_id']}"
            f"_sample_{row['sample_id']}.png"
        )

        output_path = OUTPUT_DIR / filename

        result = create_failure_visualization(
            images[index],
            masks[index],
            probability_map,
            row,
            output_path
        )

        results.append(result)

    results_df = pd.DataFrame(results)

    results_df.to_csv(
        OUTPUT_DIR / "empty_prediction_visualization_index.csv",
        index=False
    )

    print("\nVISUALIZATION SUMMARY")
    print(f"Figures generated: {len(results_df)}")

    if not results_df.empty:
        print("\nFIGURES BY CLASS")
        print(results_df["label"].value_counts().to_string())

    print(f"\nSaved in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

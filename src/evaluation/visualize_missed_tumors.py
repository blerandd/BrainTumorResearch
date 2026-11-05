
from pathlib import Path
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

from src.data.dataset_loader import BrainTumorDatasetLoader


BASE_DIR = Path(__file__).resolve().parents[2]

MODEL_PATH = BASE_DIR / "models" / "segmentation" / "unet_best.keras"

MISSED_CASES_PATH = (
    BASE_DIR / "results" / "segmentation"
    / "boundary_metrics" / "infinite_distance_cases.csv"
)

OUTPUT_DIR = (
    BASE_DIR / "results" / "segmentation"
    / "missed_tumor_analysis" / "visualizations"
)

THRESHOLD = 0.5


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
        raise ValueError("Test dataset and metadata counts do not match.")

    return images, masks, metadata


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading missed cases...")
    missed = pd.read_csv(MISSED_CASES_PATH)
    missed["sample_id"] = missed["sample_id"].astype(str)

    print("Loading model...")
    model = tf.keras.models.load_model(
        MODEL_PATH,
        compile=False
    )

    print("Loading test dataset...")
    loader = BrainTumorDatasetLoader()
    images, masks, metadata = load_test_data(loader)

    metadata["sample_id"] = metadata["sample_id"].astype(str)

    missed_ids = set(missed["sample_id"])

    indices = [
        i for i, sample_id in enumerate(metadata["sample_id"])
        if sample_id in missed_ids
    ]

    if len(indices) != len(missed):
        raise ValueError("Could not match all missed cases.")

    selected_images = images[indices]
    selected_masks = masks[indices]
    selected_metadata = metadata.iloc[indices].reset_index(drop=True)

    print(f"Generating visualizations for {len(indices)} cases...")

    predictions = model.predict(selected_images, verbose=1)

    for i in range(len(indices)):
        image = selected_images[i].squeeze()
        true_mask = selected_masks[i].squeeze() > THRESHOLD
        probability = predictions[i].squeeze()
        predicted_mask = probability >= THRESHOLD

        row = selected_metadata.iloc[i]

        sample_id = str(row["sample_id"])
        patient_id = str(row["patient_id"])
        label = str(row["label"])

        fig, axes = plt.subplots(1, 4, figsize=(17, 5))

        axes[0].imshow(image, cmap="gray")
        axes[0].set_title("Original MRI")

        axes[1].imshow(image, cmap="gray")
        axes[1].imshow(
            true_mask,
            cmap="Greens",
            alpha=0.65
        )
        axes[1].set_title("Ground Truth")

        axes[2].imshow(probability, cmap="magma", vmin=0, vmax=1)
        axes[2].set_title(
            f"Prediction Probability\nMax: {probability.max():.5f}"
        )

        axes[3].imshow(image, cmap="gray")
        axes[3].contour(
            true_mask,
            levels=[0.5],
            colors=["lime"],
            linewidths=1.5
        )

        if np.any(predicted_mask):
            axes[3].contour(
                predicted_mask,
                levels=[0.5],
                colors=["red"],
                linewidths=1.5
            )

        axes[3].set_title(
            "Overlay\nGreen: Ground Truth | Red: Prediction"
        )

        for ax in axes:
            ax.axis("off")

        fig.suptitle(
            f"Missed Tumor | Sample {sample_id} | "
            f"Patient {patient_id} | {label}",
            fontsize=13
        )

        plt.tight_layout()

        safe_id = re.sub(r"[^A-Za-z0-9_-]", "_", sample_id)

        output_path = OUTPUT_DIR / f"missed_{safe_id}.png"

        plt.savefig(
            output_path,
            dpi=150,
            bbox_inches="tight"
        )

        plt.close(fig)

    print("\nVisualization completed.")
    print(f"Saved {len(indices)} images to:")
    print(OUTPUT_DIR)


if __name__ == "__main__":
    main()

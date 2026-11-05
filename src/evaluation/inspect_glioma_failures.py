
from pathlib import Path

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


PROJECT_DIR = Path(__file__).resolve().parents[2]

METADATA_PATH = PROJECT_DIR / "data/processed/splits.csv"
MODEL_PATH = PROJECT_DIR / "models/segmentation/unet_best.keras"
METRICS_PATH = PROJECT_DIR / "results/segmentation/image_segmentation_metrics.csv"
OUTPUT_DIR = PROJECT_DIR / "results/segmentation/error_analysis"

PATIENT_IDS = ["MR031409B", "92897"]
THRESHOLD = 0.5


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    loader = BrainTumorDatasetLoader(
        processed_dir=PROJECT_DIR / "data/processed",
        image_size=(256, 256),
        batch_size=1
    )

    model = tf.keras.models.load_model(
        MODEL_PATH,
        custom_objects={
            "dice_coefficient": dice_coefficient,
            "dice_loss": dice_loss,
            "combined_segmentation_loss": combined_segmentation_loss
        },
        compile=False
    )

    metadata = pd.read_csv(METADATA_PATH)
    metrics = pd.read_csv(METRICS_PATH)

    samples = metadata[
        (metadata["patient_id"].astype(str).isin(PATIENT_IDS))
        & (metadata["split"] == "test")
        & (metadata["label"] == "glioma")
    ].copy()

    print("\nGLIOMA ERROR ANALYSIS")
    print("=" * 60)
    print(f"Selected images: {len(samples)}")

    for _, record in samples.iterrows():
        sample_id = record["sample_id"]
        patient_id = str(record["patient_id"])

        image_path = PROJECT_DIR / "data/processed" / record["image_path"]
        mask_path = PROJECT_DIR / "data/processed" / record["mask_path"]

        image = loader._load_image(str(image_path)).numpy()
        ground_truth = loader._load_mask(str(mask_path)).numpy()

        probabilities = model.predict(
            np.expand_dims(image, axis=0),
            verbose=0
        )[0]

        prediction = (probabilities >= THRESHOLD).astype(np.float32)

        true_mask = ground_truth[..., 0]
        predicted_mask = prediction[..., 0]
        probability_map = probabilities[..., 0]

        row = metrics[metrics["sample_id"] == sample_id]

        dice = float(row.iloc[0]["dice"]) if not row.empty else np.nan
        iou = float(row.iloc[0]["iou"]) if not row.empty else np.nan

        fig, axes = plt.subplots(1, 4, figsize=(18, 5))

        axes[0].imshow(image[..., 0], cmap="gray")
        axes[0].set_title("Original MRI")

        axes[1].imshow(image[..., 0], cmap="gray")
        axes[1].imshow(true_mask, cmap="Greens", alpha=0.65)
        axes[1].set_title("Ground-Truth Mask")

        axes[2].imshow(image[..., 0], cmap="gray")
        axes[2].imshow(predicted_mask, cmap="Reds", alpha=0.65)
        axes[2].set_title(f"U-Net Prediction ({THRESHOLD})")

        axes[3].imshow(image[..., 0], cmap="gray")
        axes[3].imshow(true_mask, cmap="Greens", alpha=0.45)
        axes[3].imshow(predicted_mask, cmap="Reds", alpha=0.45)
        axes[3].set_title("Ground Truth + Prediction")

        fig.suptitle(
            f"Patient {patient_id} | Sample {sample_id} | "
            f"Dice: {dice:.4f} | IoU: {iou:.4f}",
            fontsize=13
        )

        for axis in axes:
            axis.axis("off")

        plt.tight_layout()

        output_path = OUTPUT_DIR / f"patient_{patient_id}_sample_{sample_id}.png"

        plt.savefig(
            output_path,
            dpi=200,
            bbox_inches="tight"
        )

        plt.close(fig)

        print(
            f"Patient {patient_id} | Sample {sample_id} | "
            f"Dice: {dice:.4f} | Saved: {output_path.name}"
        )

    print("\nERROR ANALYSIS VISUALIZATIONS COMPLETED")
    print(f"Results saved in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

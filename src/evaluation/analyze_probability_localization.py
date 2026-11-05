
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

MODEL_PATHS = {
    "baseline": PROJECT_ROOT / "models" / "segmentation" / "unet_best.keras",
    "augmented": PROJECT_ROOT / "models" / "segmentation" / "unet_augmented_best.keras"
}

RESULTS_DIR = (
    PROJECT_ROOT
    / "results"
    / "segmentation"
    / "probability_analysis"
    / "full_test_localization"
)

THRESHOLD = 0.50
BATCH_SIZE = 8


def load_test_data():
    loader = BrainTumorDatasetLoader(
        processed_dir="data/processed",
        image_size=(256, 256),
        batch_size=16,
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

    records = (
        loader.metadata[
            loader.metadata["split"] == "test"
        ]
        .reset_index(drop=True)
    )

    if len(images) != len(records):
        raise ValueError("Test data and metadata counts do not match.")

    return images, masks, records


def load_model(path):
    if not path.is_file():
        raise FileNotFoundError(f"Model not found: {path}")

    return tf.keras.models.load_model(
        path,
        custom_objects={
            "dice_coefficient": dice_coefficient,
            "dice_loss": dice_loss,
            "combined_segmentation_loss": combined_segmentation_loss
        },
        compile=False
    )


def centroid(mask):
    coordinates = np.argwhere(mask)

    if len(coordinates) == 0:
        return None

    return coordinates.mean(axis=0)


def calculate_dice(true_mask, predicted_mask, smooth=1e-7):
    intersection = np.sum(true_mask * predicted_mask)

    return (
        2.0 * intersection + smooth
    ) / (
        np.sum(true_mask) + np.sum(predicted_mask) + smooth
    )


def analyze_image(probability, true_mask, metadata, model_name):
    true_mask = true_mask.astype(bool)
    background = ~true_mask

    tumor_probabilities = probability[true_mask]
    background_probabilities = probability[background]

    predicted = probability >= THRESHOLD

    true_center = centroid(true_mask)
    maximum_location = np.unravel_index(
        np.argmax(probability),
        probability.shape
    )

    if true_center is None:
        centroid_distance = np.nan
    else:
        centroid_distance = float(
            np.linalg.norm(
                true_center - np.array(maximum_location)
            )
        )

    tumor_mean = (
        float(np.mean(tumor_probabilities))
        if tumor_probabilities.size else np.nan
    )

    background_mean = (
        float(np.mean(background_probabilities))
        if background_probabilities.size else np.nan
    )

    tumor_max = (
        float(np.max(tumor_probabilities))
        if tumor_probabilities.size else np.nan
    )

    background_max = (
        float(np.max(background_probabilities))
        if background_probabilities.size else np.nan
    )

    true_pixels = int(np.sum(true_mask))
    predicted_pixels = int(np.sum(predicted))

    return {
        "model": model_name,
        "sample_id": metadata["sample_id"],
        "patient_id": metadata["patient_id"],
        "label": metadata["label"],
        "dice": calculate_dice(
            true_mask.astype(np.uint8),
            predicted.astype(np.uint8)
        ),
        "tumor_mean_probability": tumor_mean,
        "tumor_max_probability": tumor_max,
        "background_mean_probability": background_mean,
        "background_max_probability": background_max,
        "tumor_background_mean_gap": (
            tumor_mean - background_mean
            if np.isfinite(tumor_mean) and np.isfinite(background_mean)
            else np.nan
        ),
        "maximum_probability": float(np.max(probability)),
        "maximum_probability_location_distance": centroid_distance,
        "maximum_probability_inside_tumor": bool(
            true_mask[maximum_location]
        ),
        "predicted_pixels": predicted_pixels,
        "ground_truth_pixels": true_pixels,
        "area_ratio": (
            predicted_pixels / true_pixels
            if true_pixels > 0 else np.nan
        ),
        "empty_prediction": predicted_pixels == 0,
        "tumor_missed": (
            np.sum(predicted & true_mask) == 0
        )
    }


def evaluate_model(model_name, model_path, images, masks, records):
    print(f"\nEVALUATING {model_name.upper()}")
    print("=" * 70)

    model = load_model(model_path)

    probabilities = model.predict(
        images,
        batch_size=BATCH_SIZE,
        verbose=1
    )

    results = []

    for index in range(len(images)):
        probability = probabilities[index].squeeze()
        true_mask = masks[index].squeeze() >= 0.5

        results.append(
            analyze_image(
                probability,
                true_mask,
                records.iloc[index],
                model_name
            )
        )

    del probabilities, model
    gc.collect()

    return pd.DataFrame(results)


def summarize_results(results, group_columns):
    metrics = [
        "dice",
        "tumor_mean_probability",
        "tumor_max_probability",
        "background_mean_probability",
        "tumor_background_mean_gap",
        "maximum_probability_location_distance",
        "area_ratio"
    ]

    summary = (
        results.groupby(group_columns)
        .agg(
            image_count=("sample_id", "count"),
            **{
                f"{metric}_mean": (metric, "mean")
                for metric in metrics
            },
            empty_predictions=("empty_prediction", "sum"),
            missed_tumors=("tumor_missed", "sum")
        )
        .reset_index()
    )

    return summary


def save_probability_plot(results):
    figure, axes = plt.subplots(1, 2, figsize=(13, 5))

    for model_name, group in results.groupby("model"):
        axes[0].hist(
            group["tumor_mean_probability"].dropna(),
            bins=30,
            alpha=0.5,
            label=model_name
        )

        axes[1].hist(
            group["tumor_background_mean_gap"].dropna(),
            bins=30,
            alpha=0.5,
            label=model_name
        )

    axes[0].set_title("Mean Probability Inside Tumor")
    axes[0].set_xlabel("Mean Tumor Probability")
    axes[0].set_ylabel("Number of Images")
    axes[0].legend()

    axes[1].set_title("Tumor vs Background Probability Gap")
    axes[1].set_xlabel("Tumor Mean - Background Mean")
    axes[1].set_ylabel("Number of Images")
    axes[1].legend()

    figure.tight_layout()
    figure.savefig(
        RESULTS_DIR / "probability_localization_distributions.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.close(figure)


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("\nFULL TEST SET PROBABILITY LOCALIZATION ANALYSIS")
    print("=" * 75)

    images, masks, records = load_test_data()

    print(f"Test images: {len(images)}")
    print(f"Test patients: {records['patient_id'].nunique()}")

    baseline = evaluate_model(
        "baseline",
        MODEL_PATHS["baseline"],
        images,
        masks,
        records
    )

    augmented = evaluate_model(
        "augmented",
        MODEL_PATHS["augmented"],
        images,
        masks,
        records
    )

    results = pd.concat(
        [baseline, augmented],
        ignore_index=True
    )

    overall = summarize_results(results, ["model"])
    by_class = summarize_results(results, ["model", "label"])

    results.to_csv(
        RESULTS_DIR / "image_probability_localization.csv",
        index=False
    )

    overall.to_csv(
        RESULTS_DIR / "overall_probability_localization.csv",
        index=False
    )

    by_class.to_csv(
        RESULTS_DIR / "class_probability_localization.csv",
        index=False
    )

    save_probability_plot(results)

    print("\nOVERALL RESULTS")
    print("=" * 100)
    print(overall.to_string(index=False))

    print("\nRESULTS BY TUMOR CLASS")
    print("=" * 100)
    print(by_class.to_string(index=False))

    print("\nWORST AUGMENTED LOCALIZATION CASES")
    print("=" * 100)

    worst = (
        augmented.sort_values(
            ["tumor_mean_probability", "dice"],
            ascending=[True, True]
        )
        .head(15)
    )

    print(
        worst[
            [
                "patient_id",
                "sample_id",
                "label",
                "dice",
                "tumor_mean_probability",
                "tumor_max_probability",
                "background_max_probability",
                "maximum_probability_location_distance",
                "empty_prediction",
                "tumor_missed"
            ]
        ].to_string(index=False)
    )

    print("\nSAVED RESULTS")
    print("=" * 75)
    print(f"Image-level: {RESULTS_DIR / 'image_probability_localization.csv'}")
    print(f"Overall: {RESULTS_DIR / 'overall_probability_localization.csv'}")
    print(f"By class: {RESULTS_DIR / 'class_probability_localization.csv'}")
    print(
        f"Plot: {RESULTS_DIR / 'probability_localization_distributions.png'}"
    )

    print("\nFULL TEST SET LOCALIZATION ANALYSIS COMPLETED")


if __name__ == "__main__":
    main()

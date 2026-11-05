
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

RESULTS_DIR = (
    PROJECT_ROOT
    / "results"
    / "segmentation"
    / "failure_analysis"
)

IMAGE_THRESHOLD = 0.5
MASK_THRESHOLD = 0.5
BATCH_SIZE = 8
NUM_VISUALIZATIONS = 12


def load_test_data():
    print("\nLOADING TEST DATA")
    print("=" * 65)

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
        raise ValueError("Test image and metadata counts do not match.")

    print(f"Test images: {len(images)}")
    print(f"Unique patients: {records['patient_id'].nunique()}")
    print(f"Image shape: {images.shape}")

    return images, masks, records


def load_model(path, name):
    if not path.is_file():
        raise FileNotFoundError(f"{name} model not found: {path}")

    print(f"\nLOADING {name.upper()} MODEL")

    model = tf.keras.models.load_model(
        path,
        custom_objects={
            "dice_coefficient": dice_coefficient,
            "dice_loss": dice_loss,
            "combined_segmentation_loss": combined_segmentation_loss
        },
        compile=False
    )

    print(f"Model: {model.name}")
    print(f"Parameters: {model.count_params():,}")

    return model


def calculate_dice(y_true, y_pred, smooth=1e-7):
    y_true = y_true.astype(np.float32).ravel()
    y_pred = y_pred.astype(np.float32).ravel()

    intersection = np.sum(y_true * y_pred)

    return (
        2.0 * intersection + smooth
    ) / (
        np.sum(y_true) + np.sum(y_pred) + smooth
    )


def calculate_iou(y_true, y_pred, smooth=1e-7):
    y_true = y_true.astype(np.float32).ravel()
    y_pred = y_pred.astype(np.float32).ravel()

    intersection = np.sum(y_true * y_pred)
    union = np.sum(y_true) + np.sum(y_pred) - intersection

    return (intersection + smooth) / (union + smooth)


def classify_prediction(dice, predicted_pixels, intersection):
    if predicted_pixels == 0:
        return "empty_prediction"

    if intersection == 0:
        return "no_overlap"

    if dice < 0.5:
        return "low_overlap"

    return "acceptable_overlap"


def evaluate_model(model, model_name, images, masks, records):
    print(f"\nANALYZING {model_name.upper()} PREDICTIONS")
    print("=" * 65)

    probabilities = model.predict(
        images,
        batch_size=BATCH_SIZE,
        verbose=1
    )

    predictions = (
        probabilities >= MASK_THRESHOLD
    ).astype(np.uint8)

    ground_truth = (
        masks >= IMAGE_THRESHOLD
    ).astype(np.uint8)

    results = []

    for index in range(len(images)):
        true_mask = ground_truth[index].squeeze()
        predicted_mask = predictions[index].squeeze()
        probability_map = probabilities[index].squeeze()

        true_flat = true_mask.ravel()
        predicted_flat = predicted_mask.ravel()

        tp = int(np.sum((true_flat == 1) & (predicted_flat == 1)))
        fp = int(np.sum((true_flat == 0) & (predicted_flat == 1)))
        fn = int(np.sum((true_flat == 1) & (predicted_flat == 0)))

        true_pixels = int(np.sum(true_mask))
        predicted_pixels = int(np.sum(predicted_mask))

        dice = calculate_dice(true_mask, predicted_mask)
        iou = calculate_iou(true_mask, predicted_mask)

        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0

        inside_probabilities = probability_map[true_mask == 1]
        outside_probabilities = probability_map[true_mask == 0]

        metadata = records.iloc[index]

        results.append({
            "model": model_name,
            "sample_id": metadata["sample_id"],
            "patient_id": metadata["patient_id"],
            "label": metadata["label"],
            "dice": dice,
            "iou": iou,
            "precision": precision,
            "recall": recall,
            "true_positive_pixels": tp,
            "false_positive_pixels": fp,
            "false_negative_pixels": fn,
            "ground_truth_pixels": true_pixels,
            "predicted_pixels": predicted_pixels,
            "predicted_to_true_area_ratio": (
                predicted_pixels / true_pixels
                if true_pixels > 0 else np.nan
            ),
            "mean_probability_inside_tumor": (
                float(np.mean(inside_probabilities))
                if len(inside_probabilities) else np.nan
            ),
            "max_probability_inside_tumor": (
                float(np.max(inside_probabilities))
                if len(inside_probabilities) else np.nan
            ),
            "mean_probability_outside_tumor": (
                float(np.mean(outside_probabilities))
                if len(outside_probabilities) else np.nan
            ),
            "maximum_probability": float(np.max(probability_map)),
            "prediction_empty": predicted_pixels == 0,
            "failure_category": classify_prediction(
                dice,
                predicted_pixels,
                tp
            )
        })

    results = pd.DataFrame(results)

    save_failure_visualizations(
        images,
        ground_truth,
        predictions,
        results,
        model_name
    )

    del probabilities, predictions, ground_truth
    gc.collect()

    return results


def save_failure_visualizations(
    images,
    ground_truth,
    predictions,
    results,
    model_name
):
    output_dir = RESULTS_DIR / "visualizations" / model_name
    output_dir.mkdir(parents=True, exist_ok=True)

    ordered = results.sort_values(
        ["dice", "sample_id"]
    ).head(NUM_VISUALIZATIONS)

    for _, row in ordered.iterrows():
        index = results.index[
            results["sample_id"] == row["sample_id"]
        ][0]

        image = images[index].squeeze()
        true_mask = ground_truth[index].squeeze()
        predicted_mask = predictions[index].squeeze()

        figure, axes = plt.subplots(
            1, 5, figsize=(18, 4)
        )

        axes[0].imshow(image, cmap="gray")
        axes[0].set_title("Original MRI")

        axes[1].imshow(true_mask, cmap="gray")
        axes[1].set_title("Ground Truth")

        axes[2].imshow(predicted_mask, cmap="gray")
        axes[2].set_title("Prediction")

        axes[3].imshow(image, cmap="gray")
        axes[3].imshow(true_mask, cmap="Greens", alpha=0.4)
        axes[3].imshow(predicted_mask, cmap="Reds", alpha=0.4)
        axes[3].set_title("Green: GT | Red: Prediction")

        axes[4].imshow(
            np.abs(true_mask.astype(float) - predicted_mask.astype(float)),
            cmap="gray"
        )
        axes[4].set_title("Segmentation Difference")

        for axis in axes:
            axis.axis("off")

        figure.suptitle(
            f"{model_name.upper()} | "
            f"Patient {row['patient_id']} | "
            f"Sample {row['sample_id']} | "
            f"Class: {row['label']} | "
            f"Dice: {row['dice']:.4f} | "
            f"Failure: {row['failure_category']}"
        )

        figure.tight_layout()

        filename = (
            f"patient_{row['patient_id']}_"
            f"sample_{row['sample_id']}.png"
        )

        figure.savefig(
            output_dir / filename,
            dpi=200,
            bbox_inches="tight"
        )

        plt.close(figure)


def create_failure_summary(results):
    return (
        results
        .groupby(["model", "failure_category"])
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )


def create_class_summary(results):
    return (
        results
        .groupby(["model", "label"])
        .agg(
            image_count=("sample_id", "count"),
            dice_mean=("dice", "mean"),
            dice_median=("dice", "median"),
            empty_predictions=("prediction_empty", "sum"),
            false_positive_pixels=("false_positive_pixels", "sum"),
            false_negative_pixels=("false_negative_pixels", "sum"),
            mean_area_ratio=("predicted_to_true_area_ratio", "mean")
        )
        .reset_index()
    )


def create_paired_failure_comparison(results):
    baseline = (
        results[results["model"] == "baseline"]
        .set_index("sample_id")
    )

    augmented = (
        results[results["model"] == "augmented"]
        .set_index("sample_id")
    )

    if not baseline.index.equals(augmented.index):
        raise ValueError("Model test samples do not match.")

    paired = pd.DataFrame({
        "patient_id": baseline["patient_id"],
        "label": baseline["label"],
        "baseline_dice": baseline["dice"],
        "augmented_dice": augmented["dice"],
        "baseline_failure": baseline["failure_category"],
        "augmented_failure": augmented["failure_category"],
        "baseline_empty": baseline["prediction_empty"],
        "augmented_empty": augmented["prediction_empty"],
        "baseline_fp": baseline["false_positive_pixels"],
        "augmented_fp": augmented["false_positive_pixels"],
        "baseline_fn": baseline["false_negative_pixels"],
        "augmented_fn": augmented["false_negative_pixels"],
        "dice_change": augmented["dice"] - baseline["dice"]
    })

    return paired.reset_index()


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("\nBASELINE VS AUGMENTED FAILURE ANALYSIS")
    print("=" * 75)

    images, masks, records = load_test_data()

    baseline_model = load_model(
        BASELINE_MODEL_PATH,
        "baseline"
    )

    baseline_results = evaluate_model(
        baseline_model,
        "baseline",
        images,
        masks,
        records
    )

    del baseline_model
    gc.collect()

    augmented_model = load_model(
        AUGMENTED_MODEL_PATH,
        "augmented"
    )

    augmented_results = evaluate_model(
        augmented_model,
        "augmented",
        images,
        masks,
        records
    )

    del augmented_model
    gc.collect()

    results = pd.concat(
        [baseline_results, augmented_results],
        ignore_index=True
    )

    failure_summary = create_failure_summary(results)
    class_summary = create_class_summary(results)
    paired = create_paired_failure_comparison(results)

    print("\nFAILURE CATEGORIES")
    print("=" * 75)
    print(failure_summary.to_string(index=False))

    print("\nRESULTS BY TUMOR CLASS")
    print("=" * 75)
    print(class_summary.to_string(index=False))

    print("\nPAIRED FAILURE CHANGES")
    print("=" * 75)

    print(
        paired[
            [
                "baseline_failure",
                "augmented_failure"
            ]
        ].value_counts().to_string()
    )

    print("\nWORST AUGMENTED CASES")
    print("=" * 75)

    print(
        augmented_results
        .sort_values("dice")
        [
            [
                "patient_id",
                "sample_id",
                "label",
                "dice",
                "predicted_pixels",
                "ground_truth_pixels",
                "maximum_probability",
                "failure_category"
            ]
        ]
        .head(15)
        .to_string(index=False)
    )

    results.to_csv(
        RESULTS_DIR / "failure_analysis_image_level.csv",
        index=False
    )

    failure_summary.to_csv(
        RESULTS_DIR / "failure_category_summary.csv",
        index=False
    )

    class_summary.to_csv(
        RESULTS_DIR / "failure_analysis_by_class.csv",
        index=False
    )

    paired.to_csv(
        RESULTS_DIR / "paired_failure_comparison.csv",
        index=False
    )

    print("\nSAVED FILES")
    print("=" * 75)
    print(f"Image-level results: {RESULTS_DIR / 'failure_analysis_image_level.csv'}")
    print(f"Failure summary: {RESULTS_DIR / 'failure_category_summary.csv'}")
    print(f"Class summary: {RESULTS_DIR / 'failure_analysis_by_class.csv'}")
    print(f"Paired comparison: {RESULTS_DIR / 'paired_failure_comparison.csv'}")
    print(f"Visualizations: {RESULTS_DIR / 'visualizations'}")

    print("\nFAILURE ANALYSIS COMPLETED SUCCESSFULLY")


if __name__ == "__main__":
    main()

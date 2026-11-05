
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
from src.evaluation.boundary_metrics import (
    calculate_hd95,
    calculate_assd,
    calculate_boundary_dice
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
    / "baseline_vs_augmented"
)

IMAGE_THRESHOLD = 0.5
MASK_THRESHOLD = 0.5
BOUNDARY_TOLERANCE = 2.0
PREDICTION_BATCH_SIZE = 8


def create_directories():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def load_test_data():
    print("\nLOADING TEST DATA")
    print("=" * 65)

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

    test_records = (
        loader.metadata[
            loader.metadata["split"] == "test"
        ]
        .reset_index(drop=True)
    )

    if len(images) != len(test_records):
        raise ValueError(
            "Test image count does not match test metadata."
        )

    print(f"Test images: {len(images)}")
    print(f"Unique patients: {test_records['patient_id'].nunique()}")
    print(f"Image shape: {images.shape}")
    print(f"Mask shape: {masks.shape}")

    return images, masks, test_records


def load_model(model_path, model_name):
    if not model_path.is_file():
        raise FileNotFoundError(
            f"{model_name} model not found: {model_path}"
        )

    print(f"\nLoading {model_name} model...")
    print(f"Path: {model_path.relative_to(PROJECT_ROOT)}")

    model = tf.keras.models.load_model(
        model_path,
        custom_objects={
            "dice_coefficient": dice_coefficient,
            "dice_loss": dice_loss,
            "combined_segmentation_loss": combined_segmentation_loss
        },
        compile=False
    )

    print(f"Model name: {model.name}")
    print(f"Input shape: {model.input_shape}")
    print(f"Output shape: {model.output_shape}")
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


def evaluate_model(model, model_name, images, masks, test_records):
    print(f"\nEVALUATING {model_name.upper()} MODEL")
    print("=" * 65)

    probabilities = model.predict(
        images,
        batch_size=PREDICTION_BATCH_SIZE,
        verbose=1
    )

    predictions = (
        probabilities >= MASK_THRESHOLD
    ).astype(np.uint8)

    ground_truth = (
        masks >= IMAGE_THRESHOLD
    ).astype(np.uint8)

    records = []

    for index in range(len(images)):
        true_mask = ground_truth[index].squeeze()
        predicted_mask = predictions[index].squeeze()

        dice = calculate_dice(true_mask, predicted_mask)
        iou = calculate_iou(true_mask, predicted_mask)

        true_flat = true_mask.ravel()
        predicted_flat = predicted_mask.ravel()

        true_positive = np.sum(
            (true_flat == 1) & (predicted_flat == 1)
        )

        false_positive = np.sum(
            (true_flat == 0) & (predicted_flat == 1)
        )

        false_negative = np.sum(
            (true_flat == 1) & (predicted_flat == 0)
        )

        precision = (
            true_positive / (true_positive + false_positive)
            if true_positive + false_positive > 0
            else 0.0
        )

        recall = (
            true_positive / (true_positive + false_negative)
            if true_positive + false_negative > 0
            else 0.0
        )

        hd95 = calculate_hd95(
            true_mask,
            predicted_mask
        )

        assd = calculate_assd(
            true_mask,
            predicted_mask
        )

        boundary_dice = calculate_boundary_dice(
            true_mask,
            predicted_mask,
            tolerance=BOUNDARY_TOLERANCE
        )

        metadata = test_records.iloc[index]

        records.append({
            "model": model_name,
            "sample_id": metadata["sample_id"],
            "patient_id": metadata["patient_id"],
            "label": metadata["label"],
            "dice": dice,
            "iou": iou,
            "precision": precision,
            "recall": recall,
            "hd95_pixels": hd95,
            "assd_pixels": assd,
            "boundary_dice": boundary_dice,
            "ground_truth_pixels": int(np.sum(true_mask)),
            "predicted_pixels": int(np.sum(predicted_mask)),
            "ground_truth_empty": bool(np.sum(true_mask) == 0),
            "prediction_empty": bool(np.sum(predicted_mask) == 0)
        })

    del probabilities, predictions, ground_truth
    gc.collect()

    results = pd.DataFrame(records)

    print(f"\n{model_name} evaluation completed.")
    print(f"Evaluated images: {len(results)}")

    return results


def summarize_group(group):
    result = {
        "image_count": len(group),
        "dice_mean": group["dice"].mean(),
        "dice_std": group["dice"].std(),
        "iou_mean": group["iou"].mean(),
        "precision_mean": group["precision"].mean(),
        "recall_mean": group["recall"].mean(),
        "boundary_dice_mean": group["boundary_dice"].mean()
    }

    for metric in ["hd95_pixels", "assd_pixels"]:
        values = group[metric].to_numpy(dtype=float)
        finite_values = values[np.isfinite(values)]

        result[f"{metric}_mean_finite"] = (
            np.mean(finite_values)
            if len(finite_values) > 0
            else np.nan
        )

        result[f"{metric}_median_finite"] = (
            np.median(finite_values)
            if len(finite_values) > 0
            else np.nan
        )

        result[f"{metric}_infinite_count"] = int(
            np.sum(~np.isfinite(values))
        )

    return pd.Series(result)


def create_overall_comparison(results):
    summaries = []

    for model_name, group in results.groupby("model"):
        summary = summarize_group(group)
        summary["model"] = model_name
        summaries.append(summary)

    return pd.DataFrame(summaries).set_index("model").reset_index()


def create_class_comparison(results):
    summaries = []

    for (model_name, label), group in results.groupby(
        ["model", "label"]
    ):
        summary = summarize_group(group)
        summary["model"] = model_name
        summary["label"] = label
        summaries.append(summary)

    return pd.DataFrame(summaries)


def create_patient_comparison(results):
    summaries = []

    for (model_name, patient_id), group in results.groupby(
        ["model", "patient_id"]
    ):
        summary = summarize_group(group)
        summary["model"] = model_name
        summary["patient_id"] = patient_id
        summary["label"] = group["label"].iloc[0]
        summaries.append(summary)

    return pd.DataFrame(summaries)


def create_paired_comparison(results):
    metrics = [
        "dice",
        "iou",
        "precision",
        "recall",
        "hd95_pixels",
        "assd_pixels",
        "boundary_dice"
    ]

    baseline = (
        results[results["model"] == "baseline"]
        .set_index("sample_id")
    )

    augmented = (
        results[results["model"] == "augmented"]
        .set_index("sample_id")
    )

    if not baseline.index.equals(augmented.index):
        raise ValueError(
            "Baseline and augmented results do not contain "
            "the same test samples in the same order."
        )

    paired = pd.DataFrame({
        "patient_id": baseline["patient_id"],
        "label": baseline["label"]
    })

    for metric in metrics:
        paired[f"baseline_{metric}"] = baseline[metric]
        paired[f"augmented_{metric}"] = augmented[metric]
        paired[f"delta_{metric}"] = (
            augmented[metric] - baseline[metric]
        )

    return paired.reset_index()


def save_comparison_plot(overall):
    metrics = [
        ("dice_mean", "Dice"),
        ("iou_mean", "IoU"),
        ("precision_mean", "Precision"),
        ("recall_mean", "Recall"),
        ("boundary_dice_mean", "Boundary Dice")
    ]

    baseline = overall[
        overall["model"] == "baseline"
    ].iloc[0]

    augmented = overall[
        overall["model"] == "augmented"
    ].iloc[0]

    labels = [item[1] for item in metrics]
    baseline_values = [baseline[item[0]] for item in metrics]
    augmented_values = [augmented[item[0]] for item in metrics]

    positions = np.arange(len(labels))
    width = 0.35

    figure, axis = plt.subplots(figsize=(11, 6))

    axis.bar(
        positions - width / 2,
        baseline_values,
        width,
        label="Baseline U-Net"
    )

    axis.bar(
        positions + width / 2,
        augmented_values,
        width,
        label="Augmented U-Net"
    )

    axis.set_title("Baseline vs Augmented U-Net")
    axis.set_ylabel("Score")
    axis.set_xticks(positions)
    axis.set_xticklabels(labels)
    axis.set_ylim(0, 1)
    axis.legend()
    axis.grid(axis="y", alpha=0.3)

    figure.tight_layout()

    figure.savefig(
        RESULTS_DIR / "baseline_vs_augmented_comparison.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(figure)


def print_results(overall, class_comparison):
    print("\nOVERALL BASELINE VS AUGMENTED RESULTS")
    print("=" * 100)
    print(overall.to_string(index=False))

    print("\nRESULTS BY TUMOR CLASS")
    print("=" * 100)
    print(class_comparison.to_string(index=False))


def main():
    create_directories()

    print("\nBASELINE VS AUGMENTED U-NET EVALUATION")
    print("=" * 75)

    images, masks, test_records = load_test_data()

    baseline_model = load_model(
        BASELINE_MODEL_PATH,
        "Baseline"
    )

    baseline_results = evaluate_model(
        baseline_model,
        "baseline",
        images,
        masks,
        test_records
    )

    del baseline_model
    gc.collect()

    augmented_model = load_model(
        AUGMENTED_MODEL_PATH,
        "Augmented"
    )

    augmented_results = evaluate_model(
        augmented_model,
        "augmented",
        images,
        masks,
        test_records
    )

    del augmented_model
    gc.collect()

    results = pd.concat(
        [baseline_results, augmented_results],
        ignore_index=True
    )

    overall = create_overall_comparison(results)
    class_comparison = create_class_comparison(results)
    patient_comparison = create_patient_comparison(results)
    paired_comparison = create_paired_comparison(results)

    print_results(overall, class_comparison)

    results.to_csv(
        RESULTS_DIR / "image_level_results.csv",
        index=False
    )

    overall.to_csv(
        RESULTS_DIR / "overall_comparison.csv",
        index=False
    )

    class_comparison.to_csv(
        RESULTS_DIR / "class_comparison.csv",
        index=False
    )

    patient_comparison.to_csv(
        RESULTS_DIR / "patient_comparison.csv",
        index=False
    )

    paired_comparison.to_csv(
        RESULTS_DIR / "paired_image_comparison.csv",
        index=False
    )

    save_comparison_plot(overall)

    print("\nSAVED RESULTS")
    print("=" * 75)
    print(f"Image-level results: {RESULTS_DIR / 'image_level_results.csv'}")
    print(f"Overall comparison: {RESULTS_DIR / 'overall_comparison.csv'}")
    print(f"Class comparison: {RESULTS_DIR / 'class_comparison.csv'}")
    print(f"Patient comparison: {RESULTS_DIR / 'patient_comparison.csv'}")
    print(f"Paired comparison: {RESULTS_DIR / 'paired_image_comparison.csv'}")
    print(
        f"Comparison plot: "
        f"{RESULTS_DIR / 'baseline_vs_augmented_comparison.png'}"
    )

    print("\nBOUNDARY DISTANCE NOTES")
    print("=" * 75)
    print("HD95 and ASSD are reported in 256x256 model pixels.")
    print("Finite means exclude infinite cases; infinite counts are reported separately.")
    print(f"Boundary Dice tolerance: {BOUNDARY_TOLERANCE} pixels.")

    print("\nBASELINE VS AUGMENTED EVALUATION COMPLETED")


if __name__ == "__main__":
    main()

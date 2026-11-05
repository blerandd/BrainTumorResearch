
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

MODEL_PATHS = {
    "baseline": PROJECT_ROOT / "models/segmentation/unet_best.keras",
    "flip_only": PROJECT_ROOT / "models/segmentation/unet_flip_only_best.keras",
    "flip_rotation": PROJECT_ROOT / "models/segmentation/unet_augmented_best.keras"
}

RESULTS_DIR = (
    PROJECT_ROOT
    / "results"
    / "segmentation"
    / "three_model_comparison"
)

PREDICTION_BATCH_SIZE = 8
MASK_THRESHOLD = 0.5
BOUNDARY_TOLERANCE = 2.0


def create_directories():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)


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

    test_records = (
        loader.metadata[
            loader.metadata["split"] == "test"
        ]
        .reset_index(drop=True)
    )

    if len(images) != len(test_records):
        raise ValueError("Test images do not match metadata.")

    print(f"Test images: {len(images)}")
    print(f"Test patients: {test_records['patient_id'].nunique()}")
    print(f"Image shape: {images.shape}")
    print(f"Mask shape: {masks.shape}")

    return images, masks, test_records


def load_model(model_name, model_path):
    if not model_path.is_file():
        raise FileNotFoundError(
            f"Missing {model_name} model: {model_path}"
        )

    print(f"\nLOADING {model_name.upper()} MODEL")
    print("=" * 70)

    model = tf.keras.models.load_model(
        model_path,
        custom_objects={
            "dice_coefficient": dice_coefficient,
            "dice_loss": dice_loss,
            "combined_segmentation_loss": combined_segmentation_loss
        },
        compile=False
    )

    print(f"Model: {model.name}")
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


def evaluate_model(model_name, model, images, masks, metadata):
    print(f"\nEVALUATING {model_name.upper()}")
    print("=" * 70)

    probabilities = model.predict(
        images,
        batch_size=PREDICTION_BATCH_SIZE,
        verbose=1
    )

    predictions = (
        probabilities >= MASK_THRESHOLD
    ).astype(np.uint8)

    ground_truth = (
        masks >= MASK_THRESHOLD
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

        hd95 = calculate_hd95(true_mask, predicted_mask)
        assd = calculate_assd(true_mask, predicted_mask)

        boundary_dice = calculate_boundary_dice(
            true_mask,
            predicted_mask,
            tolerance=BOUNDARY_TOLERANCE
        )

        record = metadata.iloc[index]

        records.append({
            "model": model_name,
            "sample_id": record["sample_id"],
            "patient_id": record["patient_id"],
            "label": record["label"],
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

    print(f"{model_name}: evaluated {len(results)} images.")

    return results


def summarize_group(group):
    summary = {
        "image_count": len(group),
        "dice_mean": group["dice"].mean(),
        "dice_std": group["dice"].std(),
        "dice_median": group["dice"].median(),
        "iou_mean": group["iou"].mean(),
        "precision_mean": group["precision"].mean(),
        "recall_mean": group["recall"].mean(),
        "boundary_dice_mean": group["boundary_dice"].mean()
    }

    for metric in ["hd95_pixels", "assd_pixels"]:
        values = group[metric].to_numpy(dtype=float)
        finite_values = values[np.isfinite(values)]

        summary[f"{metric}_mean_finite"] = (
            np.mean(finite_values)
            if len(finite_values) else np.nan
        )

        summary[f"{metric}_median_finite"] = (
            np.median(finite_values)
            if len(finite_values) else np.nan
        )

        summary[f"{metric}_infinite_count"] = int(
            np.sum(~np.isfinite(values))
        )

    return pd.Series(summary)


def create_overall_comparison(results):
    rows = []

    for model_name, group in results.groupby("model"):
        row = summarize_group(group).to_dict()
        row["model"] = model_name
        rows.append(row)

    return pd.DataFrame(rows)


def create_class_comparison(results):
    rows = []

    for (model_name, label), group in results.groupby(
        ["model", "label"]
    ):
        row = summarize_group(group).to_dict()
        row["model"] = model_name
        row["label"] = label
        rows.append(row)

    return pd.DataFrame(rows)


def create_patient_comparison(results):
    rows = []

    for (model_name, patient_id), group in results.groupby(
        ["model", "patient_id"]
    ):
        row = summarize_group(group).to_dict()
        row["model"] = model_name
        row["patient_id"] = patient_id
        row["label"] = group["label"].iloc[0]
        rows.append(row)

    return pd.DataFrame(rows)


def create_paired_image_comparison(results):
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
        .sort_index()
    )

    if baseline.empty:
        raise ValueError("Baseline results are missing.")

    paired = baseline[["patient_id", "label"]].copy()

    for model_name in ["flip_only", "flip_rotation"]:
        current = (
            results[results["model"] == model_name]
            .set_index("sample_id")
            .sort_index()
        )

        if not baseline.index.equals(current.index):
            raise ValueError(
                f"Test samples do not match for {model_name}."
            )

        for metric in metrics:
            paired[f"{model_name}_{metric}"] = current[metric]
            paired[f"baseline_{metric}"] = baseline[metric]
            paired[f"{model_name}_delta_{metric}"] = (
                current[metric] - baseline[metric]
            )

    return paired.reset_index()


def create_paired_patient_comparison(patient_results):
    metrics = [
        "dice_mean",
        "iou_mean",
        "precision_mean",
        "recall_mean",
        "boundary_dice_mean"
    ]

    rows = []

    for patient_id, group in patient_results.groupby("patient_id"):
        row = {
            "patient_id": patient_id,
            "label": group["label"].iloc[0]
        }

        indexed = group.set_index("model")

        for model_name in ["baseline", "flip_only", "flip_rotation"]:
            if model_name not in indexed.index:
                raise ValueError(
                    f"Missing {model_name} for patient {patient_id}."
                )

            for metric in metrics:
                row[f"{model_name}_{metric}"] = (
                    indexed.loc[model_name, metric]
                )

        for model_name in ["flip_only", "flip_rotation"]:
            row[f"{model_name}_dice_change"] = (
                row[f"{model_name}_dice_mean"]
                - row["baseline_dice_mean"]
            )

        rows.append(row)

    return pd.DataFrame(rows)


def save_comparison_plots(overall, class_comparison):
    model_order = ["baseline", "flip_only", "flip_rotation"]

    overall = (
        overall.set_index("model")
        .reindex(model_order)
        .reset_index()
    )

    metrics = [
        ("dice_mean", "Dice"),
        ("iou_mean", "IoU"),
        ("precision_mean", "Precision"),
        ("recall_mean", "Recall"),
        ("boundary_dice_mean", "Boundary Dice")
    ]

    figure, axis = plt.subplots(figsize=(12, 6))

    positions = np.arange(len(metrics))
    width = 0.25

    for index, model_name in enumerate(model_order):
        row = overall[overall["model"] == model_name].iloc[0]
        values = [row[column] for column, _ in metrics]

        axis.bar(
            positions + (index - 1) * width,
            values,
            width,
            label=model_name.replace("_", " ").title()
        )

    axis.set_title("Three-Model U-Net Comparison")
    axis.set_ylabel("Score")
    axis.set_xticks(positions)
    axis.set_xticklabels([name for _, name in metrics])
    axis.set_ylim(0, 1)
    axis.legend()
    axis.grid(axis="y", alpha=0.3)

    figure.tight_layout()
    figure.savefig(
        RESULTS_DIR / "three_model_overall_comparison.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.close(figure)

    dice_by_class = class_comparison.pivot(
        index="label",
        columns="model",
        values="dice_mean"
    ).reindex(columns=model_order)

    axis = dice_by_class.plot(
        kind="bar",
        figsize=(10, 6)
    )

    axis.set_title("Dice Comparison by Tumor Class")
    axis.set_xlabel("Tumor class")
    axis.set_ylabel("Mean Dice")
    axis.set_ylim(0, 1)
    axis.grid(axis="y", alpha=0.3)
    axis.legend(title="Model")

    figure = axis.get_figure()
    figure.tight_layout()
    figure.savefig(
        RESULTS_DIR / "three_model_class_comparison.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.close(figure)


def print_results(overall, class_comparison, patient_comparison):
    print("\nOVERALL THREE-MODEL COMPARISON")
    print("=" * 110)
    print(overall.to_string(index=False))

    print("\nCOMPARISON BY TUMOR CLASS")
    print("=" * 110)
    print(
        class_comparison[
            [
                "model",
                "label",
                "image_count",
                "dice_mean",
                "iou_mean",
                "precision_mean",
                "recall_mean",
                "boundary_dice_mean",
                "hd95_pixels_mean_finite",
                "assd_pixels_mean_finite"
            ]
        ].to_string(index=False)
    )

    print("\nPATIENT-LEVEL COMPARISON")
    print("=" * 110)
    print(
        patient_comparison[
            [
                "model",
                "patient_id",
                "label",
                "image_count",
                "dice_mean"
            ]
        ].to_string(index=False)
    )


def main():
    create_directories()

    print("\nTHREE-MODEL U-NET EVALUATION")
    print("=" * 75)

    images, masks, test_records = load_test_data()

    all_results = []

    for model_name, model_path in MODEL_PATHS.items():
        model = load_model(model_name, model_path)

        model_results = evaluate_model(
            model_name,
            model,
            images,
            masks,
            test_records
        )

        all_results.append(model_results)

        del model
        gc.collect()

        tf.keras.backend.clear_session()

    results = pd.concat(
        all_results,
        ignore_index=True
    )

    overall = create_overall_comparison(results)
    class_comparison = create_class_comparison(results)
    patient_comparison = create_patient_comparison(results)

    paired_images = create_paired_image_comparison(results)
    paired_patients = create_paired_patient_comparison(
        patient_comparison
    )

    print_results(
        overall,
        class_comparison,
        patient_comparison
    )

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

    paired_images.to_csv(
        RESULTS_DIR / "paired_image_comparison.csv",
        index=False
    )

    paired_patients.to_csv(
        RESULTS_DIR / "paired_patient_comparison.csv",
        index=False
    )

    save_comparison_plots(
        overall,
        class_comparison
    )

    print("\nSAVED RESULTS")
    print("=" * 75)
    print(f"Directory: {RESULTS_DIR}")

    for filename in [
        "image_level_results.csv",
        "overall_comparison.csv",
        "class_comparison.csv",
        "patient_comparison.csv",
        "paired_image_comparison.csv",
        "paired_patient_comparison.csv",
        "three_model_overall_comparison.png",
        "three_model_class_comparison.png"
    ]:
        print(RESULTS_DIR / filename)

    print("\nMETRIC NOTES")
    print("=" * 75)
    print("HD95 and ASSD are measured in 256x256 model pixels.")
    print("Finite distance means exclude infinite cases.")
    print(f"Boundary Dice tolerance: {BOUNDARY_TOLERANCE} pixels.")
    print("All models were evaluated on the same test samples.")

    print("\nTHREE-MODEL EVALUATION COMPLETED SUCCESSFULLY")


if __name__ == "__main__":
    main()

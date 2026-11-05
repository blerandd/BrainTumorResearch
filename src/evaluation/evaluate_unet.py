
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.metrics import precision_score, recall_score

from src.data.dataset_loader import BrainTumorDatasetLoader
from src.models.unet import dice_coefficient, dice_loss, combined_segmentation_loss


PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = PROJECT_ROOT / "models" / "segmentation" / "unet_best.keras"

RESULTS_DIR = PROJECT_ROOT / "results" / "segmentation"

IMAGE_THRESHOLD = 0.5
MASK_THRESHOLD = 0.5
NUM_VISUALIZATIONS = 12


def find_model():
    if MODEL_PATH.is_file():
        return MODEL_PATH

    model_candidates = list(
        (PROJECT_ROOT / "models").rglob("*.keras")
    )

    model_candidates = [
        path for path in model_candidates
        if "unet" in path.name.lower()
    ]

    if len(model_candidates) == 1:
        return model_candidates[0]

    if not model_candidates:
        raise FileNotFoundError(
            "No U-Net .keras model was found. "
            "Check where train_unet.py saves the trained model."
        )

    candidates = "\n".join(
        str(path.relative_to(PROJECT_ROOT))
        for path in model_candidates
    )

    raise FileNotFoundError(
        "Multiple U-Net models were found. "
        "Set MODEL_PATH to the correct model:\n" + candidates
    )


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


def calculate_metrics(y_true, y_pred):
    true_flat = y_true.astype(np.uint8).ravel()
    pred_flat = y_pred.astype(np.uint8).ravel()

    dice = calculate_dice(true_flat, pred_flat)
    iou = calculate_iou(true_flat, pred_flat)

    precision = precision_score(
        true_flat,
        pred_flat,
        zero_division=0
    )

    recall = recall_score(
        true_flat,
        pred_flat,
        zero_division=0
    )

    return {
        "dice": dice,
        "iou": iou,
        "precision": precision,
        "recall": recall
    }


def load_test_data(loader):
    datasets = loader.get_segmentation_datasets()
    test_dataset = datasets["test"]

    images = []
    masks = []

    for batch_images, batch_masks in test_dataset:
        images.append(batch_images.numpy())
        masks.append(batch_masks.numpy())

    if not images:
        raise ValueError("The test dataset is empty.")

    images = np.concatenate(images, axis=0)
    masks = np.concatenate(masks, axis=0)

    test_records = (
        loader.metadata[
            loader.metadata["split"] == "test"
        ]
        .reset_index(drop=True)
    )

    if len(test_records) != len(images):
        raise ValueError(
            "Test metadata and loaded image counts do not match."
        )

    return images, masks, test_records


def evaluate_model(model, images, masks, test_records):
    print("\nGENERATING TEST PREDICTIONS")
    print("=" * 65)

    probabilities = model.predict(
        images,
        batch_size=8,
        verbose=1
    )

    predictions = (
        probabilities >= MASK_THRESHOLD
    ).astype(np.uint8)

    ground_truth = (
        masks >= IMAGE_THRESHOLD
    ).astype(np.uint8)

    records = []

    print("\nCALCULATING IMAGE-LEVEL METRICS")
    print("=" * 65)

    for index in range(len(images)):
        metrics = calculate_metrics(
            ground_truth[index],
            predictions[index]
        )

        records.append({
            "sample_id": test_records.iloc[index]["sample_id"],
            "patient_id": test_records.iloc[index]["patient_id"],
            "label": test_records.iloc[index]["label"],
            "dice": metrics["dice"],
            "iou": metrics["iou"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "ground_truth_pixels": int(
                np.sum(ground_truth[index])
            ),
            "predicted_pixels": int(
                np.sum(predictions[index])
            )
        })

    return probabilities, predictions, ground_truth, pd.DataFrame(records)


def create_summary(image_metrics):
    metric_columns = [
        "dice",
        "iou",
        "precision",
        "recall"
    ]

    summary = []

    for metric in metric_columns:
        summary.append({
            "metric": metric,
            "mean": image_metrics[metric].mean(),
            "std": image_metrics[metric].std(),
            "median": image_metrics[metric].median(),
            "minimum": image_metrics[metric].min(),
            "maximum": image_metrics[metric].max()
        })

    return pd.DataFrame(summary)


def create_class_summary(image_metrics):
    return (
        image_metrics
        .groupby("label")
        .agg(
            number_of_images=("sample_id", "count"),
            dice_mean=("dice", "mean"),
            dice_std=("dice", "std"),
            iou_mean=("iou", "mean"),
            iou_std=("iou", "std"),
            precision_mean=("precision", "mean"),
            recall_mean=("recall", "mean")
        )
        .reset_index()
    )


def create_patient_summary(image_metrics):
    return (
        image_metrics
        .groupby("patient_id")
        .agg(
            number_of_images=("sample_id", "count"),
            dice_mean=("dice", "mean"),
            iou_mean=("iou", "mean"),
            precision_mean=("precision", "mean"),
            recall_mean=("recall", "mean")
        )
        .reset_index()
    )


def save_visualizations(
    images,
    ground_truth,
    predictions,
    image_metrics
):
    visualization_dir = RESULTS_DIR / "visualizations"
    visualization_dir.mkdir(parents=True, exist_ok=True)

    count = min(NUM_VISUALIZATIONS, len(images))

    ordered_indices = (
        image_metrics["dice"]
        .sort_values()
        .index
        .to_numpy()
    )

    if count == 1:
        selected_indices = ordered_indices[:1]
    else:
        selected_indices = np.unique(
            np.linspace(
                0,
                len(ordered_indices) - 1,
                count,
                dtype=int
            )
        )

        selected_indices = ordered_indices[selected_indices]

    for index in selected_indices:
        image = images[index].squeeze()
        true_mask = ground_truth[index].squeeze()
        predicted_mask = predictions[index].squeeze()

        row = image_metrics.iloc[index]

        figure, axes = plt.subplots(
            1,
            4,
            figsize=(16, 4)
        )

        axes[0].imshow(image, cmap="gray")
        axes[0].set_title("Original MRI")

        axes[1].imshow(true_mask, cmap="gray")
        axes[1].set_title("Ground-Truth Mask")

        axes[2].imshow(predicted_mask, cmap="gray")
        axes[2].set_title("Predicted Mask")

        axes[3].imshow(image, cmap="gray")
        axes[3].imshow(
            true_mask,
            cmap="Greens",
            alpha=0.35
        )
        axes[3].imshow(
            predicted_mask,
            cmap="Reds",
            alpha=0.35
        )
        axes[3].set_title("Overlay: Green = True, Red = Predicted")

        for axis in axes:
            axis.axis("off")

        figure.suptitle(
            f"Sample {row['sample_id']} | "
            f"Patient {row['patient_id']} | "
            f"Class: {row['label']} | "
            f"Dice: {row['dice']:.4f} | "
            f"IoU: {row['iou']:.4f}"
        )

        figure.tight_layout()

        filename = (
            f"sample_{row['sample_id']}_"
            f"patient_{row['patient_id']}.png"
        )

        figure.savefig(
            visualization_dir / filename,
            dpi=200,
            bbox_inches="tight"
        )

        plt.close(figure)


def save_metric_plots(image_metrics):
    figure, axes = plt.subplots(
        2,
        2,
        figsize=(12, 9)
    )

    metrics = [
        ("dice", "Dice Coefficient"),
        ("iou", "Intersection over Union"),
        ("precision", "Pixel Precision"),
        ("recall", "Pixel Recall")
    ]

    for axis, (column, title) in zip(axes.flat, metrics):
        axis.hist(
            image_metrics[column],
            bins=20,
            edgecolor="black"
        )

        axis.axvline(
            image_metrics[column].mean(),
            linestyle="--",
            label=f"Mean: {image_metrics[column].mean():.4f}"
        )

        axis.set_title(title)
        axis.set_xlabel("Score")
        axis.set_ylabel("Number of Images")
        axis.legend()
        axis.grid(alpha=0.3)

    figure.suptitle("U-Net Test Set Segmentation Metrics")
    figure.tight_layout()

    figure.savefig(
        RESULTS_DIR / "segmentation_metrics_distribution.png",
        dpi=200,
        bbox_inches="tight"
    )

    plt.close(figure)


def save_class_plot(class_summary):
    figure, axis = plt.subplots(figsize=(9, 5))

    axis.bar(
        class_summary["label"],
        class_summary["dice_mean"],
        yerr=class_summary["dice_std"].fillna(0),
        capsize=5
    )

    axis.set_title("U-Net Dice Score by Tumor Class")
    axis.set_xlabel("Tumor Class")
    axis.set_ylabel("Mean Dice Coefficient")
    axis.set_ylim(0, 1)
    axis.grid(axis="y", alpha=0.3)

    figure.tight_layout()

    figure.savefig(
        RESULTS_DIR / "dice_by_class.png",
        dpi=200,
        bbox_inches="tight"
    )

    plt.close(figure)


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("\nU-NET SEGMENTATION EVALUATION")
    print("=" * 65)

    model_path = find_model()

    print(f"Model: {model_path.relative_to(PROJECT_ROOT)}")

    print("\nLOADING TRAINED U-NET")
    print("=" * 65)

    model = tf.keras.models.load_model(
        model_path,
        custom_objects={
            "dice_coefficient": dice_coefficient,
            "dice_loss": dice_loss,
            "combined_segmentation_loss": combined_segmentation_loss
        },
        compile=False
    )

    print(f"Model loaded: {model.name}")
    print(f"Input shape: {model.input_shape}")
    print(f"Output shape: {model.output_shape}")
    print(f"Parameters: {model.count_params():,}")

    print("\nLOADING TEST DATA")
    print("=" * 65)

    loader = BrainTumorDatasetLoader()

    images, masks, test_records = load_test_data(loader)

    print(f"Test images: {len(images)}")
    print(f"Unique test patients: {test_records['patient_id'].nunique()}")
    print(f"Image shape: {images.shape}")
    print(f"Mask shape: {masks.shape}")

    probabilities, predictions, ground_truth, image_metrics = (
        evaluate_model(
            model,
            images,
            masks,
            test_records
        )
    )

    summary = create_summary(image_metrics)
    class_summary = create_class_summary(image_metrics)
    patient_summary = create_patient_summary(image_metrics)

    print("\nOVERALL TEST RESULTS")
    print("=" * 65)
    print(summary.to_string(index=False))

    print("\nRESULTS BY TUMOR CLASS")
    print("=" * 65)
    print(class_summary.to_string(index=False))

    print("\nPATIENT-LEVEL SEGMENTATION SUMMARY")
    print("=" * 65)
    print(
        patient_summary[
            ["dice_mean", "iou_mean"]
        ].mean().to_string()
    )

    image_metrics.to_csv(
        RESULTS_DIR / "image_segmentation_metrics.csv",
        index=False
    )

    summary.to_csv(
        RESULTS_DIR / "segmentation_summary.csv",
        index=False
    )

    class_summary.to_csv(
        RESULTS_DIR / "segmentation_by_class.csv",
        index=False
    )

    patient_summary.to_csv(
        RESULTS_DIR / "patient_segmentation_metrics.csv",
        index=False
    )

    save_visualizations(
        images,
        ground_truth,
        predictions,
        image_metrics
    )

    save_metric_plots(image_metrics)
    save_class_plot(class_summary)

    print("\nSAVED EVALUATION FILES")
    print("=" * 65)
    print(f"Image metrics: {RESULTS_DIR / 'image_segmentation_metrics.csv'}")
    print(f"Overall summary: {RESULTS_DIR / 'segmentation_summary.csv'}")
    print(f"Class summary: {RESULTS_DIR / 'segmentation_by_class.csv'}")
    print(f"Patient summary: {RESULTS_DIR / 'patient_segmentation_metrics.csv'}")
    print(f"Visualizations: {RESULTS_DIR / 'visualizations'}")
    print(f"Metric distribution: {RESULTS_DIR / 'segmentation_metrics_distribution.png'}")
    print(f"Dice by class: {RESULTS_DIR / 'dice_by_class.png'}")

    print("\nU-NET SEGMENTATION EVALUATION COMPLETED SUCCESSFULLY")


if __name__ == "__main__":
    main()

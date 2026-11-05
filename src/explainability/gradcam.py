
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

from src.data.dataset_loader import BrainTumorDatasetLoader


SEED = 42
IMAGE_SIZE = (256, 256)
BATCH_SIZE = 16

MODEL_PATH = Path("models/classification/custom_cnn_best.keras")
OUTPUT_DIR = Path("results/explainability/gradcam_improved")

CLASSES = ["glioma", "meningioma", "pituitary"]
TARGET_LAYER_NAME = "activation_3"
HEATMAP_PERCENTILES = [50, 70, 80, 90]


def create_directories():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def create_gradcam_model(model):
    target_layer = model.get_layer(TARGET_LAYER_NAME)

    print(f"Grad-CAM target layer: {target_layer.name}")

    return tf.keras.Model(
        inputs=model.inputs,
        outputs=[
            target_layer.output,
            model.output
        ]
    )


def generate_gradcam(gradcam_model, image, class_index):
    image_batch = tf.expand_dims(image, axis=0)

    with tf.GradientTape() as tape:
        feature_maps, predictions = gradcam_model(
            image_batch,
            training=False
        )

        class_score = predictions[:, class_index]

    gradients = tape.gradient(
        class_score,
        feature_maps
    )

    if gradients is None:
        raise RuntimeError(
            "Grad-CAM gradients could not be calculated."
        )

    weights = tf.reduce_mean(
        gradients,
        axis=(1, 2)
    )

    feature_maps = feature_maps[0]
    weights = weights[0]

    heatmap = tf.reduce_sum(
        feature_maps * weights,
        axis=-1
    )

    heatmap = tf.nn.relu(heatmap)

    heatmap = tf.image.resize(
        heatmap[..., tf.newaxis],
        IMAGE_SIZE,
        method="bilinear"
    )

    heatmap = tf.squeeze(heatmap)

    maximum = tf.reduce_max(heatmap)

    heatmap = tf.math.divide_no_nan(
        heatmap,
        maximum
    )

    return heatmap.numpy()


def load_mask(loader, relative_path):
    full_path = loader.processed_dir / relative_path

    mask = loader._load_mask(str(full_path))

    return tf.squeeze(mask).numpy()


def calculate_overlap(heatmap, mask, percentile):
    threshold = np.percentile(
        heatmap,
        percentile
    )

    predicted_region = heatmap >= threshold
    ground_truth = mask > 0.5

    intersection = np.logical_and(
        predicted_region,
        ground_truth
    ).sum()

    union = np.logical_or(
        predicted_region,
        ground_truth
    ).sum()

    predicted_area = predicted_region.sum()
    ground_truth_area = ground_truth.sum()

    dice = (
        2.0 * intersection /
        (predicted_area + ground_truth_area)
        if predicted_area + ground_truth_area > 0
        else 0.0
    )

    iou = (
        intersection / union
        if union > 0
        else 0.0
    )

    return float(dice), float(iou)


def save_visualization(
    image,
    true_heatmap,
    predicted_heatmap,
    mask,
    true_label,
    predicted_label,
    confidence,
    sample_id,
    patient_id,
    correct
):
    figure, axes = plt.subplots(
        1,
        5,
        figsize=(21, 5)
    )

    axes[0].imshow(
        image,
        cmap="gray"
    )
    axes[0].set_title("Original MRI")

    axes[1].imshow(
        true_heatmap,
        cmap="jet",
        vmin=0,
        vmax=1
    )
    axes[1].set_title(
        f"Actual Class: {true_label}"
    )

    axes[2].imshow(
        predicted_heatmap,
        cmap="jet",
        vmin=0,
        vmax=1
    )
    axes[2].set_title(
        f"Predicted Class: {predicted_label}"
    )

    axes[3].imshow(
        image,
        cmap="gray"
    )
    axes[3].imshow(
        predicted_heatmap,
        cmap="jet",
        alpha=0.4,
        vmin=0,
        vmax=1
    )
    axes[3].set_title("Predicted-Class Overlay")

    axes[4].imshow(
        image,
        cmap="gray"
    )
    axes[4].contour(
        mask,
        levels=[0.5],
        colors=["lime"],
        linewidths=1.5
    )
    axes[4].set_title("Ground-Truth Mask")

    for axis in axes:
        axis.axis("off")

    status = "Correct" if correct else "Incorrect"

    figure.suptitle(
        f"{status} | True: {true_label} | "
        f"Predicted: {predicted_label} | "
        f"Confidence: {confidence:.3f} | "
        f"Patient: {patient_id}",
        fontsize=12
    )

    figure.tight_layout()

    category = "correct" if correct else "incorrect"

    output_path = (
        OUTPUT_DIR /
        f"{category}_sample_{sample_id}_patient_{patient_id}.png"
    )

    figure.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(figure)

    return output_path


def main():
    tf.keras.utils.set_random_seed(SEED)

    create_directories()

    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}"
        )

    print("\nLOADING TRAINED CNN")
    print("=" * 60)

    model = tf.keras.models.load_model(
        MODEL_PATH,
        compile=False
    )

    gradcam_model = create_gradcam_model(model)

    print("\nLOADING TEST DATA")
    print("=" * 60)

    loader = BrainTumorDatasetLoader(
        processed_dir="data/processed",
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        seed=SEED
    )

    test_records = loader.metadata[
        loader.metadata["split"] == "test"
    ].copy().reset_index(drop=True)

    predictions = []

    for start in range(0, len(test_records), BATCH_SIZE):
        batch_records = test_records.iloc[
            start:start + BATCH_SIZE
        ]

        images = []

        for relative_path in batch_records["image_path"]:
            full_path = loader.processed_dir / relative_path

            image = loader._load_image(str(full_path))

            images.append(image.numpy())

        image_batch = np.stack(images)

        batch_predictions = model.predict(
            image_batch,
            verbose=0
        )

        predictions.extend(batch_predictions)

    predictions = np.asarray(predictions)

    true_indices = np.asarray([
        loader.class_to_index[label]
        for label in test_records["label"]
    ])

    predicted_indices = np.argmax(
        predictions,
        axis=1
    )

    test_records["true_label"] = [
        CLASSES[index] for index in true_indices
    ]

    test_records["predicted_label"] = [
        CLASSES[index] for index in predicted_indices
    ]

    test_records["confidence"] = np.max(
        predictions,
        axis=1
    )

    test_records["correct"] = (
        true_indices == predicted_indices
    )

    print(f"Test images: {len(test_records)}")
    print(
        f"Correct predictions: "
        f"{test_records['correct'].sum()}"
    )
    print(
        f"Incorrect predictions: "
        f"{(~test_records['correct']).sum()}"
    )

    selected_indices = []

    for class_name in CLASSES:
        class_records = test_records[
            test_records["true_label"] == class_name
        ]

        correct_records = class_records[
            class_records["correct"]
        ].head(2)

        incorrect_records = class_records[
            ~class_records["correct"]
        ].head(2)

        selected_indices.extend(
            correct_records.index.tolist()
        )

        selected_indices.extend(
            incorrect_records.index.tolist()
        )

    selected_indices = sorted(set(selected_indices))

    print(
        f"\nSelected images: {len(selected_indices)}"
    )

    results = []

    print("\nGENERATING IMPROVED GRAD-CAM")
    print("=" * 60)

    for record_index in selected_indices:
        record = test_records.iloc[record_index]

        image_path = (
            loader.processed_dir / record["image_path"]
        )

        image = loader._load_image(str(image_path))

        image_array = tf.squeeze(image).numpy()

        true_label = record["true_label"]
        predicted_label = record["predicted_label"]

        true_class_index = CLASSES.index(true_label)
        predicted_class_index = CLASSES.index(predicted_label)

        true_heatmap = generate_gradcam(
            gradcam_model,
            image,
            true_class_index
        )

        predicted_heatmap = generate_gradcam(
            gradcam_model,
            image,
            predicted_class_index
        )

        mask = load_mask(
            loader,
            record["mask_path"]
        )

        overlap_results = {}

        for percentile in HEATMAP_PERCENTILES:
            true_dice, true_iou = calculate_overlap(
                true_heatmap,
                mask,
                percentile
            )

            predicted_dice, predicted_iou = calculate_overlap(
                predicted_heatmap,
                mask,
                percentile
            )

            overlap_results[
                f"actual_class_dice_p{percentile}"
            ] = true_dice

            overlap_results[
                f"actual_class_iou_p{percentile}"
            ] = true_iou

            overlap_results[
                f"predicted_class_dice_p{percentile}"
            ] = predicted_dice

            overlap_results[
                f"predicted_class_iou_p{percentile}"
            ] = predicted_iou

        output_path = save_visualization(
            image=image_array,
            true_heatmap=true_heatmap,
            predicted_heatmap=predicted_heatmap,
            mask=mask,
            true_label=true_label,
            predicted_label=predicted_label,
            confidence=float(record["confidence"]),
            sample_id=record["sample_id"],
            patient_id=record["patient_id"],
            correct=bool(record["correct"])
        )

        result = {
            "sample_id": record["sample_id"],
            "patient_id": record["patient_id"],
            "true_label": true_label,
            "predicted_label": predicted_label,
            "confidence": float(record["confidence"]),
            "correct": bool(record["correct"]),
            **overlap_results,
            "visualization_path": str(output_path)
        }

        results.append(result)

        print(
            f"Sample {record['sample_id']} | "
            f"True: {true_label} | "
            f"Predicted: {predicted_label} | "
            f"Correct: {record['correct']}"
        )

    results_df = pd.DataFrame(results)

    csv_path = OUTPUT_DIR / "gradcam_analysis.csv"

    results_df.to_csv(
        csv_path,
        index=False
    )

    print("\nGRAD-CAM ANALYSIS")
    print("=" * 60)

    print(
        results_df[
            [
                "sample_id",
                "true_label",
                "predicted_label",
                "correct",
                "actual_class_dice_p80",
                "predicted_class_dice_p80"
            ]
        ].to_string(index=False)
    )

    print("\nSAVED RESULTS")
    print("=" * 60)
    print(f"Visualizations: {OUTPUT_DIR}")
    print(f"Analysis CSV: {csv_path}")

    print("\nIMPROVED GRAD-CAM COMPLETED SUCCESSFULLY")


if __name__ == "__main__":
    main()

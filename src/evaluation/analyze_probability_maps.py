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
    / "probability_analysis"
)

THRESHOLDS = [0.10, 0.25, 0.50, 0.75]

SELECTED_CASES = [
    {
        "patient_id": "MR049104B",
        "sample_id": "1912",
        "description": "Empty prediction"
    },
    {
        "patient_id": "MR051644C",
        "sample_id": "2596",
        "description": "Incorrect localization"
    }
]


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

    if not images:
        raise ValueError("Test dataset is empty.")

    images = np.concatenate(images, axis=0)
    masks = np.concatenate(masks, axis=0)

    records = (
        loader.metadata[
            loader.metadata["split"] == "test"
        ]
        .reset_index(drop=True)
    )

    if len(images) != len(records):
        raise ValueError(
            "Test image count does not match metadata count."
        )

    print(f"Test images: {len(images)}")
    print(f"Test patients: {records['patient_id'].nunique()}")
    print(f"Image shape: {images.shape}")
    print(f"Mask shape: {masks.shape}")

    return images, masks, records


def load_model(path, name):
    if not path.is_file():
        raise FileNotFoundError(
            f"{name} model not found: {path}"
        )

    print(f"\nLOADING {name.upper()} MODEL")
    print("=" * 70)

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
    print(f"Input shape: {model.input_shape}")
    print(f"Parameters: {model.count_params():,}")

    return model


def calculate_dice(true_mask, predicted_mask, smooth=1e-7):
    true_mask = true_mask.astype(np.float32).ravel()
    predicted_mask = predicted_mask.astype(np.float32).ravel()

    intersection = np.sum(true_mask * predicted_mask)

    return (
        2.0 * intersection + smooth
    ) / (
        np.sum(true_mask) + np.sum(predicted_mask) + smooth
    )


def calculate_centroid(mask):
    coordinates = np.argwhere(mask > 0)

    if len(coordinates) == 0:
        return None

    return coordinates.mean(axis=0)


def calculate_centroid_distance(true_mask, predicted_mask):
    true_centroid = calculate_centroid(true_mask)
    predicted_centroid = calculate_centroid(predicted_mask)

    if true_centroid is None or predicted_centroid is None:
        return np.inf

    return float(
        np.linalg.norm(true_centroid - predicted_centroid)
    )


def calculate_probability_statistics(probability_map, true_mask):
    tumor_probabilities = probability_map[true_mask > 0]
    background_probabilities = probability_map[true_mask == 0]

    return {
        "maximum_probability": float(np.max(probability_map)),
        "mean_probability": float(np.mean(probability_map)),
        "median_probability": float(np.median(probability_map)),
        "maximum_tumor_probability": (
            float(np.max(tumor_probabilities))
            if len(tumor_probabilities) else np.nan
        ),
        "mean_tumor_probability": (
            float(np.mean(tumor_probabilities))
            if len(tumor_probabilities) else np.nan
        ),
        "mean_background_probability": (
            float(np.mean(background_probabilities))
            if len(background_probabilities) else np.nan
        )
    }


def evaluate_thresholds(
    probability_map,
    true_mask,
    model_name,
    patient_id,
    sample_id,
    label
):
    records = []

    for threshold in THRESHOLDS:
        predicted_mask = (
            probability_map >= threshold
        ).astype(np.uint8)

        records.append({
            "model": model_name,
            "patient_id": patient_id,
            "sample_id": sample_id,
            "label": label,
            "threshold": threshold,
            "dice": calculate_dice(
                true_mask,
                predicted_mask
            ),
            "predicted_pixels": int(
                np.sum(predicted_mask)
            ),
            "ground_truth_pixels": int(
                np.sum(true_mask)
            ),
            "area_ratio": (
                float(np.sum(predicted_mask) / np.sum(true_mask))
                if np.sum(true_mask) > 0
                else np.nan
            ),
            "centroid_distance_pixels": calculate_centroid_distance(
                true_mask,
                predicted_mask
            ),
            "maximum_probability": float(
                np.max(probability_map)
            ),
            "mean_tumor_probability": (
                float(np.mean(probability_map[true_mask > 0]))
                if np.any(true_mask > 0)
                else np.nan
            )
        })

    return records


def save_case_visualization(
    image,
    true_mask,
    baseline_probability,
    augmented_probability,
    case,
    label
):
    patient_id = case["patient_id"]
    sample_id = case["sample_id"]

    baseline_prediction = (
        baseline_probability >= 0.50
    ).astype(np.uint8)

    augmented_prediction = (
        augmented_probability >= 0.50
    ).astype(np.uint8)

    figure, axes = plt.subplots(
        2,
        4,
        figsize=(17, 9)
    )

    axes[0, 0].imshow(image, cmap="gray")
    axes[0, 0].set_title("Original MRI")

    axes[0, 1].imshow(true_mask, cmap="gray")
    axes[0, 1].set_title("Ground Truth")

    baseline_heatmap = axes[0, 2].imshow(
        baseline_probability,
        cmap="inferno",
        vmin=0,
        vmax=1
    )
    axes[0, 2].set_title("Baseline Probability Map")
    figure.colorbar(
        baseline_heatmap,
        ax=axes[0, 2],
        fraction=0.046
    )

    augmented_heatmap = axes[0, 3].imshow(
        augmented_probability,
        cmap="inferno",
        vmin=0,
        vmax=1
    )
    axes[0, 3].set_title("Augmented Probability Map")
    figure.colorbar(
        augmented_heatmap,
        ax=axes[0, 3],
        fraction=0.046
    )

    axes[1, 0].imshow(baseline_prediction, cmap="gray")
    axes[1, 0].set_title("Baseline Prediction (0.50)")

    axes[1, 1].imshow(augmented_prediction, cmap="gray")
    axes[1, 1].set_title("Augmented Prediction (0.50)")

    axes[1, 2].imshow(image, cmap="gray")
    axes[1, 2].imshow(
        true_mask,
        cmap="Greens",
        alpha=0.40
    )
    axes[1, 2].imshow(
        baseline_prediction,
        cmap="Reds",
        alpha=0.40
    )
    axes[1, 2].set_title("Baseline Overlay")

    axes[1, 3].imshow(image, cmap="gray")
    axes[1, 3].imshow(
        true_mask,
        cmap="Greens",
        alpha=0.40
    )
    axes[1, 3].imshow(
        augmented_prediction,
        cmap="Blues",
        alpha=0.40
    )
    axes[1, 3].set_title("Augmented Overlay")

    for axis in axes.flat:
        axis.axis("off")

    figure.suptitle(
        f"Patient {patient_id} | Sample {sample_id} | "
        f"Class: {label} | {case['description']}",
        fontsize=13
    )

    figure.tight_layout()

    output_path = (
        RESULTS_DIR
        / "visualizations"
        / f"patient_{patient_id}_sample_{sample_id}_probability.png"
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    figure.savefig(
        output_path,
        dpi=250,
        bbox_inches="tight"
    )

    plt.close(figure)

    print(f"Saved visualization: {output_path}")


def save_threshold_plot(threshold_results):
    for case_key, group in threshold_results.groupby(
        ["patient_id", "sample_id"]
    ):
        patient_id, sample_id = case_key

        figure, axis = plt.subplots(figsize=(8, 5))

        for model_name, model_group in group.groupby("model"):
            axis.plot(
                model_group["threshold"],
                model_group["dice"],
                marker="o",
                linewidth=2,
                label=model_name.capitalize()
            )

        axis.set_title(
            f"Dice vs Threshold - Patient {patient_id}, "
            f"Sample {sample_id}"
        )
        axis.set_xlabel("Probability Threshold")
        axis.set_ylabel("Dice Coefficient")
        axis.set_xticks(THRESHOLDS)
        axis.set_ylim(0, 1)
        axis.grid(alpha=0.3)
        axis.legend()

        figure.tight_layout()

        output_path = (
            RESULTS_DIR
            / "threshold_plots"
            / f"patient_{patient_id}_sample_{sample_id}_thresholds.png"
        )

        output_path.parent.mkdir(parents=True, exist_ok=True)

        figure.savefig(
            output_path,
            dpi=250,
            bbox_inches="tight"
        )

        plt.close(figure)


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("\nPROBABILITY MAP ANALYSIS")
    print("=" * 70)

    images, masks, records = load_test_data()

    selected_indices = {}

    for case in SELECTED_CASES:
        matches = records[
            (records["patient_id"].astype(str) == case["patient_id"])
            & (records["sample_id"].astype(str) == case["sample_id"])
        ]

        if len(matches) != 1:
            raise ValueError(
                f"Expected exactly one test record for "
                f"{case['patient_id']} / {case['sample_id']}, "
                f"found {len(matches)}."
            )

        selected_indices[
            (case["patient_id"], case["sample_id"])
        ] = int(matches.index[0])

    baseline_model = load_model(
        BASELINE_MODEL_PATH,
        "Baseline"
    )

    augmented_model = load_model(
        AUGMENTED_MODEL_PATH,
        "Augmented"
    )

    all_threshold_records = []

    for case in SELECTED_CASES:
        key = (case["patient_id"], case["sample_id"])
        index = selected_indices[key]

        image = images[index].squeeze()
        true_mask = (
            masks[index].squeeze() >= 0.5
        ).astype(np.uint8)

        metadata = records.iloc[index]
        label = metadata["label"]

        print("\n" + "=" * 70)
        print(
            f"PATIENT {case['patient_id']} | "
            f"SAMPLE {case['sample_id']} | CLASS {label}"
        )
        print("=" * 70)

        selected_image = np.expand_dims(
            images[index],
            axis=0
        )

        baseline_probability = baseline_model.predict(
            selected_image,
            verbose=0
        )[0].squeeze()

        augmented_probability = augmented_model.predict(
            selected_image,
            verbose=0
        )[0].squeeze()

        for model_name, probability_map in [
            ("baseline", baseline_probability),
            ("augmented", augmented_probability)
        ]:
            statistics = calculate_probability_statistics(
                probability_map,
                true_mask
            )

            print(f"\n{model_name.upper()} PROBABILITY STATISTICS")
            print("-" * 50)

            for metric, value in statistics.items():
                print(f"{metric}: {value:.8f}")

            threshold_records = evaluate_thresholds(
                probability_map,
                true_mask,
                model_name,
                case["patient_id"],
                case["sample_id"],
                label
            )

            all_threshold_records.extend(threshold_records)

        save_case_visualization(
            image,
            true_mask,
            baseline_probability,
            augmented_probability,
            case,
            label
        )

        del baseline_probability, augmented_probability
        gc.collect()

    del baseline_model, augmented_model
    gc.collect()

    threshold_results = pd.DataFrame(all_threshold_records)

    threshold_path = (
        RESULTS_DIR / "probability_threshold_results.csv"
    )

    threshold_results.to_csv(
        threshold_path,
        index=False
    )

    save_threshold_plot(threshold_results)

    print("\nTHRESHOLD ANALYSIS")
    print("=" * 90)
    print(
        threshold_results[
            [
                "patient_id",
                "sample_id",
                "model",
                "threshold",
                "dice",
                "predicted_pixels",
                "area_ratio",
                "centroid_distance_pixels"
            ]
        ].to_string(index=False)
    )

    print("\nSAVED FILES")
    print("=" * 70)
    print(f"Threshold results: {threshold_path}")
    print(f"Visualizations: {RESULTS_DIR / 'visualizations'}")
    print(f"Threshold plots: {RESULTS_DIR / 'threshold_plots'}")

    print("\nPROBABILITY MAP ANALYSIS COMPLETED SUCCESSFULLY")


if __name__ == "__main__":
    main()
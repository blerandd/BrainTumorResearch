
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

from src.data.dataset_loader import BrainTumorDatasetLoader
from src.evaluation.evaluate_boundary_metrics import (
    MODEL_PATH,
    MASK_THRESHOLD,
    load_test_data
)


BASE_DIR = Path(__file__).resolve().parents[2]

OUTPUT_DIR = (
    BASE_DIR
    / "results"
    / "segmentation"
    / "diagnostics"
    / "empty_prediction_confidence"
)

DIAGNOSTIC_THRESHOLDS = [0.10, 0.25, 0.40, 0.50]


def calculate_region_statistics(probabilities, region):
    values = probabilities[region]

    if values.size == 0:
        return {
            "mean": np.nan,
            "max": np.nan,
            "p95": np.nan,
            "p99": np.nan
        }

    return {
        "mean": float(np.mean(values)),
        "max": float(np.max(values)),
        "p95": float(np.percentile(values, 95)),
        "p99": float(np.percentile(values, 99))
    }


def analyze_predictions(model, images, masks, metadata):
    predictions = model.predict(images, verbose=1)

    results = []

    for index in range(len(images)):
        probabilities = predictions[index].squeeze()
        true_mask = masks[index].squeeze() > MASK_THRESHOLD

        predicted_mask = probabilities > MASK_THRESHOLD

        ground_truth_empty = not np.any(true_mask)
        prediction_empty = not np.any(predicted_mask)

        tumor_stats = calculate_region_statistics(
            probabilities,
            true_mask
        )

        background_stats = calculate_region_statistics(
            probabilities,
            ~true_mask
        )

        row = metadata.iloc[index]

        result = {
            "sample_id": row["sample_id"],
            "patient_id": row["patient_id"],
            "label": row["label"],
            "ground_truth_empty": ground_truth_empty,
            "prediction_empty": prediction_empty,
            "max_probability": float(np.max(probabilities)),
            "mean_probability": float(np.mean(probabilities)),
            "p95_probability": float(np.percentile(probabilities, 95)),
            "p99_probability": float(np.percentile(probabilities, 99)),
            "tumor_mean_probability": tumor_stats["mean"],
            "tumor_max_probability": tumor_stats["max"],
            "tumor_p95_probability": tumor_stats["p95"],
            "tumor_p99_probability": tumor_stats["p99"],
            "background_mean_probability": background_stats["mean"],
            "background_max_probability": background_stats["max"],
            "tumor_pixel_count": int(np.sum(true_mask)),
            "predicted_pixel_count": int(np.sum(predicted_mask))
        }

        for threshold in DIAGNOSTIC_THRESHOLDS:
            result[f"pixels_above_{threshold:.2f}"] = int(
                np.sum(probabilities > threshold)
            )

            result[f"tumor_pixels_above_{threshold:.2f}"] = int(
                np.sum((probabilities > threshold) & true_mask)
            )

        results.append(result)

    return pd.DataFrame(results)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading baseline U-Net...")

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    model = tf.keras.models.load_model(
        MODEL_PATH,
        compile=False
    )

    print("Loading test dataset...")

    loader = BrainTumorDatasetLoader()
    images, masks, metadata = load_test_data(loader)

    print(f"Test images: {len(images)}")

    print("\nAnalyzing prediction probabilities...")

    results = analyze_predictions(
        model,
        images,
        masks,
        metadata
    )

    false_negative_cases = results[
        (~results["ground_truth_empty"])
        & results["prediction_empty"]
    ].copy()

    false_negative_cases = false_negative_cases.sort_values(
        "max_probability"
    )

    all_results_path = OUTPUT_DIR / "all_prediction_confidence.csv"
    failed_results_path = OUTPUT_DIR / "empty_prediction_cases.csv"

    results.to_csv(all_results_path, index=False)
    false_negative_cases.to_csv(failed_results_path, index=False)

    print("\nPREDICTION SUMMARY")
    print(f"Total test images: {len(results)}")
    print(f"Empty predictions: {len(false_negative_cases)}")

    print("\nEMPTY PREDICTIONS BY CLASS")
    print(
        false_negative_cases.groupby("label").size().to_string()
    )

    print("\nCONFIDENCE STATISTICS FOR EMPTY PREDICTIONS")

    columns = [
        "max_probability",
        "mean_probability",
        "p95_probability",
        "p99_probability",
        "tumor_mean_probability",
        "tumor_max_probability",
        "tumor_p95_probability",
        "tumor_p99_probability"
    ]

    print(
        false_negative_cases[columns]
        .describe()
        .round(6)
        .to_string()
    )

    print("\nFAILED CASES WITH HIGHEST MAXIMUM PROBABILITY")

    print(
        false_negative_cases[
            [
                "sample_id",
                "patient_id",
                "label",
                "max_probability",
                "tumor_mean_probability",
                "tumor_max_probability",
                "tumor_p99_probability"
            ]
        ]
        .sort_values("max_probability", ascending=False)
        .head(15)
        .to_string(index=False)
    )

    print("\nSAVED RESULTS")
    print(all_results_path)
    print(failed_results_path)


if __name__ == "__main__":
    main()

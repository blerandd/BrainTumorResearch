from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]

RESULTS_DIR = ROOT / "results" / "segmentation"

IMAGE_METRICS_PATH = RESULTS_DIR / "image_segmentation_metrics.csv"
PATIENT_METRICS_PATH = RESULTS_DIR / "patient_segmentation_metrics.csv"

OUTPUT_DIR = RESULTS_DIR / "statistical_analysis"

BOOTSTRAP_ITERATIONS = 10000
RANDOM_SEED = 42
CONFIDENCE_LEVEL = 0.95

METRICS = [
    "dice_mean",
    "iou_mean",
    "precision_mean",
    "recall_mean"
]

METRIC_LABELS = {
    "dice_mean": "Dice",
    "iou_mean": "IoU",
    "precision_mean": "Precision",
    "recall_mean": "Recall"
}

CLASSES = [
    "glioma",
    "meningioma",
    "pituitary"
]


def load_patient_data():
    image_df = pd.read_csv(IMAGE_METRICS_PATH)
    patient_df = pd.read_csv(PATIENT_METRICS_PATH)

    required_image_columns = {
        "patient_id",
        "label"
    }

    required_patient_columns = {
        "patient_id",
        "number_of_images",
        *METRICS
    }

    if not required_image_columns.issubset(image_df.columns):
        raise ValueError("Image metrics CSV is missing required columns.")

    if not required_patient_columns.issubset(patient_df.columns):
        raise ValueError("Patient metrics CSV is missing required columns.")

    image_df["patient_id"] = image_df["patient_id"].astype(str)
    patient_df["patient_id"] = patient_df["patient_id"].astype(str)

    patient_classes = (
        image_df.groupby("patient_id")["label"]
        .agg(lambda values: values.unique().tolist())
    )

    inconsistent = patient_classes[
        patient_classes.apply(len) != 1
    ]

    if not inconsistent.empty:
        raise ValueError(
            "Some patients have multiple tumor labels."
        )

    class_df = patient_classes.apply(
        lambda values: values[0]
    ).rename("label").reset_index()

    patient_df = patient_df.merge(
        class_df,
        on="patient_id",
        how="left",
        validate="one_to_one"
    )

    if patient_df["label"].isnull().any():
        raise ValueError("Some patients have no matching tumor class.")

    if patient_df.duplicated("patient_id").any():
        raise ValueError("Duplicate patient records detected.")

    if patient_df[METRICS].isnull().any().any():
        raise ValueError("Missing metric values detected.")

    if not set(patient_df["label"]).issubset(CLASSES):
        raise ValueError("Unexpected tumor class detected.")

    return patient_df


def bootstrap_confidence_interval(values, rng):
    values = np.asarray(values, dtype=np.float64)

    if len(values) < 2:
        return np.nan, np.nan

    bootstrap_means = np.empty(
        BOOTSTRAP_ITERATIONS,
        dtype=np.float64
    )

    for iteration in range(BOOTSTRAP_ITERATIONS):
        sample = rng.choice(
            values,
            size=len(values),
            replace=True
        )

        bootstrap_means[iteration] = np.mean(sample)

    alpha = 1.0 - CONFIDENCE_LEVEL

    lower = np.quantile(
        bootstrap_means,
        alpha / 2
    )

    upper = np.quantile(
        bootstrap_means,
        1 - alpha / 2
    )

    return lower, upper


def analyze_group(group, scope, rng):
    results = []

    for metric in METRICS:
        values = group[metric].to_numpy(dtype=np.float64)

        lower, upper = bootstrap_confidence_interval(
            values,
            rng
        )

        results.append({
            "scope": scope,
            "metric": METRIC_LABELS[metric],
            "number_of_patients": len(values),
            "mean": np.mean(values),
            "standard_deviation": np.std(values, ddof=1),
            "median": np.median(values),
            "ci_lower": lower,
            "ci_upper": upper,
            "confidence_level": CONFIDENCE_LEVEL
        })

    return results


def run_analysis():
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    rng = np.random.default_rng(RANDOM_SEED)

    patient_df = load_patient_data()

    print("\nPATIENT-LEVEL BOOTSTRAP ANALYSIS")
    print("=" * 65)

    print(f"Total patients: {len(patient_df)}")
    print(f"Bootstrap iterations: {BOOTSTRAP_ITERATIONS}")
    print(f"Confidence level: {CONFIDENCE_LEVEL:.0%}")

    all_results = []

    overall_results = analyze_group(
        patient_df,
        "overall",
        rng
    )

    all_results.extend(overall_results)

    for tumor_class in CLASSES:
        class_patients = patient_df[
            patient_df["label"] == tumor_class
        ]

        class_results = analyze_group(
            class_patients,
            tumor_class,
            rng
        )

        all_results.extend(class_results)

    results_df = pd.DataFrame(all_results)

    output_csv = OUTPUT_DIR / "patient_bootstrap_results.csv"

    results_df.to_csv(
        output_csv,
        index=False
    )

    print("\nOVERALL RESULTS")
    print("=" * 65)

    print(
        results_df[
            results_df["scope"] == "overall"
        ].to_string(index=False)
    )

    print("\nRESULTS BY TUMOR CLASS")
    print("=" * 65)

    for tumor_class in CLASSES:
        print(f"\n{tumor_class.upper()}")

        print(
            results_df[
                results_df["scope"] == tumor_class
            ].to_string(index=False)
        )

    plot_confidence_intervals(results_df)

    print("\nSAVED RESULTS")
    print("=" * 65)
    print(f"CSV: {output_csv}")
    print(
        f"Plot: {OUTPUT_DIR / 'patient_confidence_intervals.png'}"
    )

    print("\nPATIENT-LEVEL BOOTSTRAP COMPLETED SUCCESSFULLY")


def plot_confidence_intervals(results_df):
    metrics = ["Dice", "IoU"]

    scopes = [
        "overall",
        "glioma",
        "meningioma",
        "pituitary"
    ]

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(14, 6)
    )

    for axis, metric in zip(axes, metrics):
        metric_df = results_df[
            results_df["metric"] == metric
        ].set_index("scope").reindex(scopes)

        means = metric_df["mean"].to_numpy()
        lower = metric_df["ci_lower"].to_numpy()
        upper = metric_df["ci_upper"].to_numpy()

        errors = np.vstack([
            means - lower,
            upper - means
        ])

        axis.errorbar(
            means,
            scopes,
            xerr=errors,
            fmt="o",
            capsize=5
        )

        axis.set_title(
            f"{metric} with 95% Confidence Intervals"
        )

        axis.set_xlabel(metric)
        axis.set_xlim(0, 1)
        axis.grid(axis="x", alpha=0.3)

    fig.suptitle(
        "Patient-Level U-Net Segmentation Performance"
    )

    plt.tight_layout()

    output_path = OUTPUT_DIR / "patient_confidence_intervals.png"

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)


if __name__ == "__main__":
    run_analysis()
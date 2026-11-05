
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

LOCALIZATION_PATH = (
    PROJECT_ROOT
    / "results"
    / "segmentation"
    / "probability_analysis"
    / "full_test_localization"
    / "image_probability_localization.csv"
)

PATIENT_COMPARISON_PATH = (
    PROJECT_ROOT
    / "results"
    / "segmentation"
    / "baseline_vs_augmented"
    / "patient_comparison.csv"
)

FAILURE_PATH = (
    PROJECT_ROOT
    / "results"
    / "segmentation"
    / "failure_analysis"
    / "failure_analysis_image_level.csv"
)

RESULTS_DIR = (
    PROJECT_ROOT
    / "results"
    / "segmentation"
    / "statistical_analysis"
    / "glioma_patient_failures"
)


def load_data():
    print("\nLOADING EXISTING ANALYSIS RESULTS")
    print("=" * 75)

    for path in [
        LOCALIZATION_PATH,
        PATIENT_COMPARISON_PATH,
        FAILURE_PATH
    ]:
        if not path.is_file():
            raise FileNotFoundError(f"Required file not found: {path}")

    localization = pd.read_csv(LOCALIZATION_PATH)
    patient_comparison = pd.read_csv(PATIENT_COMPARISON_PATH)
    failures = pd.read_csv(FAILURE_PATH)

    for dataframe in [
        localization,
        patient_comparison,
        failures
    ]:
        dataframe["patient_id"] = (
            dataframe["patient_id"].astype(str)
        )

    print(f"Localization records: {len(localization)}")
    print(f"Patient comparison records: {len(patient_comparison)}")
    print(f"Failure records: {len(failures)}")

    return localization, patient_comparison, failures


def create_patient_localization_summary(localization):
    metrics = [
        "dice",
        "tumor_mean_probability",
        "tumor_max_probability",
        "background_mean_probability",
        "tumor_background_mean_gap",
        "maximum_probability_location_distance",
        "area_ratio"
    ]

    aggregation = {
        "image_count": ("sample_id", "count"),
        "empty_predictions": ("empty_prediction", "sum"),
        "missed_tumors": ("tumor_missed", "sum")
    }

    for metric in metrics:
        aggregation[f"{metric}_mean"] = (metric, "mean")
        aggregation[f"{metric}_median"] = (metric, "median")

    summary = (
        localization.groupby(
            ["model", "patient_id", "label"]
        )
        .agg(**aggregation)
        .reset_index()
    )

    summary["missed_tumor_percentage"] = (
        summary["missed_tumors"]
        / summary["image_count"]
        * 100
    )

    summary["empty_prediction_percentage"] = (
        summary["empty_predictions"]
        / summary["image_count"]
        * 100
    )

    return summary


def create_paired_patient_table(patient_summary):
    metrics = [
        "dice_mean",
        "tumor_mean_probability_mean",
        "tumor_max_probability_mean",
        "tumor_background_mean_gap_mean",
        "maximum_probability_location_distance_mean",
        "missed_tumors",
        "missed_tumor_percentage",
        "empty_predictions"
    ]

    baseline = (
        patient_summary[
            patient_summary["model"] == "baseline"
        ]
        .set_index("patient_id")
    )

    augmented = (
        patient_summary[
            patient_summary["model"] == "augmented"
        ]
        .set_index("patient_id")
    )

    common_patients = baseline.index.intersection(
        augmented.index
    )

    baseline = baseline.loc[common_patients].sort_index()
    augmented = augmented.loc[common_patients].sort_index()

    paired = pd.DataFrame({
        "patient_id": common_patients,
        "label": baseline["label"].values,
        "image_count": baseline["image_count"].values
    })

    for metric in metrics:
        paired[f"baseline_{metric}"] = baseline[metric].values
        paired[f"augmented_{metric}"] = augmented[metric].values

        paired[f"change_{metric}"] = (
            paired[f"augmented_{metric}"]
            - paired[f"baseline_{metric}"]
        )

    paired["dice_improved"] = (
        paired["change_dice_mean"] > 0
    )

    paired["missed_tumors_increased"] = (
        paired["change_missed_tumors"] > 0
    )

    return paired


def merge_failure_categories(paired, failures):
    if "failure_category" not in failures.columns:
        print(
            "\nWARNING: failure_category column not found. "
            "Skipping failure-category aggregation."
        )
        return paired, None

    failure_counts = (
        failures.groupby(
            ["model", "patient_id", "failure_category"]
        )
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )

    categories = [
        column for column in failure_counts.columns
        if column not in ["model", "patient_id"]
    ]

    baseline = (
        failure_counts[
            failure_counts["model"] == "baseline"
        ]
        .drop(columns="model")
        .set_index("patient_id")
    )

    augmented = (
        failure_counts[
            failure_counts["model"] == "augmented"
        ]
        .drop(columns="model")
        .set_index("patient_id")
    )

    all_patients = paired["patient_id"].astype(str)

    baseline = baseline.reindex(
        all_patients,
        fill_value=0
    )

    augmented = augmented.reindex(
        all_patients,
        fill_value=0
    )

    for category in categories:
        paired[f"baseline_failure_{category}"] = (
            baseline[category].to_numpy()
        )

        paired[f"augmented_failure_{category}"] = (
            augmented[category].to_numpy()
        )

        paired[f"change_failure_{category}"] = (
            paired[f"augmented_failure_{category}"]
            - paired[f"baseline_failure_{category}"]
        )

    return paired, failure_counts


def analyze_glioma(paired):
    glioma = paired[
        paired["label"].str.lower() == "glioma"
    ].copy()

    if glioma.empty:
        raise ValueError("No glioma patients found.")

    print("\nGLIOMA PATIENT COMPARISON")
    print("=" * 100)

    columns = [
        "patient_id",
        "image_count",
        "baseline_dice_mean",
        "augmented_dice_mean",
        "change_dice_mean",
        "baseline_missed_tumors",
        "augmented_missed_tumors",
        "change_missed_tumors",
        "baseline_tumor_mean_probability_mean",
        "augmented_tumor_mean_probability_mean"
    ]

    print(
        glioma[columns]
        .sort_values("change_dice_mean")
        .to_string(index=False)
    )

    print("\nGLIOMA SUMMARY")
    print("=" * 75)

    print(f"Number of glioma patients: {len(glioma)}")

    print(
        "Patients with improved Dice:",
        int(glioma["dice_improved"].sum())
    )

    print(
        "Patients with increased missed tumors:",
        int(glioma["missed_tumors_increased"].sum())
    )

    print(
        "Mean patient Dice change:",
        round(glioma["change_dice_mean"].mean(), 6)
    )

    print(
        "Median patient Dice change:",
        round(glioma["change_dice_mean"].median(), 6)
    )

    return glioma


def create_patient_risk_ranking(paired):
    ranking = paired.copy()

    ranking["risk_score"] = (
        ranking["augmented_missed_tumor_percentage"].fillna(0)
        + ranking["augmented_empty_predictions"].fillna(0)
        * 2
        + ranking["augmented_maximum_probability_location_distance_mean"].fillna(0)
        / 10
    )

    ranking = ranking.sort_values(
        "risk_score",
        ascending=False
    )

    return ranking


def create_class_summary(paired):
    return (
        paired.groupby("label")
        .agg(
            patient_count=("patient_id", "count"),
            mean_baseline_dice=("baseline_dice_mean", "mean"),
            mean_augmented_dice=("augmented_dice_mean", "mean"),
            mean_dice_change=("change_dice_mean", "mean"),
            mean_baseline_missed=("baseline_missed_tumors", "mean"),
            mean_augmented_missed=("augmented_missed_tumors", "mean"),
            patients_improved=("dice_improved", "sum"),
            patients_with_more_misses=("missed_tumors_increased", "sum")
        )
        .reset_index()
    )


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    localization, patient_comparison, failures = load_data()

    print("\nCREATING PATIENT-LEVEL LOCALIZATION SUMMARY")
    print("=" * 75)

    patient_summary = create_patient_localization_summary(
        localization
    )

    paired = create_paired_patient_table(
        patient_summary
    )

    paired, failure_counts = merge_failure_categories(
        paired,
        failures
    )

    glioma = analyze_glioma(paired)

    class_summary = create_class_summary(paired)

    risk_ranking = create_patient_risk_ranking(paired)

    print("\nCLASS COMPARISON")
    print("=" * 100)
    print(class_summary.to_string(index=False))

    print("\nTOP 15 PATIENTS BY EXPLORATORY RISK SCORE")
    print("=" * 100)

    ranking_columns = [
        "patient_id",
        "label",
        "augmented_dice_mean",
        "augmented_missed_tumors",
        "augmented_missed_tumor_percentage",
        "augmented_maximum_probability_location_distance_mean",
        "risk_score"
    ]

    print(
        risk_ranking[ranking_columns]
        .head(15)
        .to_string(index=False)
    )

    patient_summary.to_csv(
        RESULTS_DIR / "patient_localization_summary.csv",
        index=False
    )

    paired.to_csv(
        RESULTS_DIR / "paired_patient_localization_comparison.csv",
        index=False
    )

    glioma.to_csv(
        RESULTS_DIR / "glioma_patient_failure_comparison.csv",
        index=False
    )

    class_summary.to_csv(
        RESULTS_DIR / "class_patient_failure_summary.csv",
        index=False
    )

    risk_ranking.to_csv(
        RESULTS_DIR / "patient_risk_ranking.csv",
        index=False
    )

    if failure_counts is not None:
        failure_counts.to_csv(
            RESULTS_DIR / "failure_category_counts.csv",
            index=False
        )

    print("\nSAVED FILES")
    print("=" * 75)

    for path in sorted(RESULTS_DIR.glob("*.csv")):
        print(path)

    print("\nGLIOMA PATIENT FAILURE ANALYSIS COMPLETED")


if __name__ == "__main__":
    main()

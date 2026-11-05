
from pathlib import Path
from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, wilcoxon


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_PATH = (
    PROJECT_ROOT
    / "results"
    / "segmentation"
    / "three_model_comparison"
    / "patient_comparison.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "segmentation"
    / "three_model_comparison"
    / "statistical_analysis"
)

MODELS = ["baseline", "flip_only", "flip_rotation"]
CLASSES = ["glioma", "meningioma", "pituitary"]

BOOTSTRAP_ITERATIONS = 10000
RANDOM_SEED = 42


def create_directories():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_patient_results():
    print("\nLOADING PATIENT-LEVEL RESULTS")
    print("=" * 70)

    if not INPUT_PATH.is_file():
        raise FileNotFoundError(
            f"Patient comparison file not found: {INPUT_PATH}"
        )

    data = pd.read_csv(INPUT_PATH)

    required_columns = {
        "model",
        "patient_id",
        "label",
        "dice_mean"
    }

    missing = required_columns - set(data.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    data = data[
        data["model"].isin(MODELS)
    ].copy()

    if data.duplicated(["model", "patient_id"]).any():
        raise ValueError(
            "Duplicate model-patient records found."
        )

    wide = data.pivot(
        index="patient_id",
        columns="model",
        values="dice_mean"
    )

    wide = wide.reindex(columns=MODELS)

    metadata = (
        data[["patient_id", "label"]]
        .drop_duplicates()
        .set_index("patient_id")
    )

    if metadata.index.duplicated().any():
        raise ValueError(
            "A patient has inconsistent tumor-class labels."
        )

    wide = wide.join(metadata)

    incomplete = wide[MODELS].isnull().any(axis=1)

    if incomplete.any():
        missing_patients = wide.index[incomplete].tolist()

        raise ValueError(
            "Some patients do not have results for all three models: "
            f"{missing_patients}"
        )

    wide = wide.reset_index()

    print(f"Patients analyzed: {len(wide)}")
    print(f"Models: {', '.join(MODELS)}")

    print("\nPATIENT COUNTS BY CLASS")
    print(wide.groupby("label")["patient_id"].nunique().to_string())

    return wide


def bootstrap_mean_difference(
    differences,
    iterations=BOOTSTRAP_ITERATIONS,
    seed=RANDOM_SEED
):
    values = np.asarray(differences, dtype=float)
    values = values[np.isfinite(values)]

    if len(values) == 0:
        return np.nan, np.nan, np.nan

    rng = np.random.default_rng(seed)

    bootstrap_means = np.empty(iterations)

    for index in range(iterations):
        sample = rng.choice(
            values,
            size=len(values),
            replace=True
        )

        bootstrap_means[index] = np.mean(sample)

    lower, upper = np.percentile(
        bootstrap_means,
        [2.5, 97.5]
    )

    return np.mean(values), lower, upper


def holm_adjust(p_values):
    p_values = np.asarray(p_values, dtype=float)

    adjusted = np.full(len(p_values), np.nan)
    valid_indices = np.where(np.isfinite(p_values))[0]

    if len(valid_indices) == 0:
        return adjusted

    valid_p = p_values[valid_indices]
    order = np.argsort(valid_p)

    sorted_p = valid_p[order]
    count = len(sorted_p)

    adjusted_sorted = np.maximum.accumulate(
        [
            min(1.0, (count - rank) * p_value)
            for rank, p_value in enumerate(sorted_p)
        ]
    )

    adjusted_valid = np.empty(count)
    adjusted_valid[order] = adjusted_sorted

    adjusted[valid_indices] = adjusted_valid

    return adjusted


def run_friedman(group, analysis_name):
    arrays = [
        group[model].to_numpy(dtype=float)
        for model in MODELS
    ]

    if len(group) < 3:
        return {
            "analysis": analysis_name,
            "patient_count": len(group),
            "friedman_statistic": np.nan,
            "friedman_p_value": np.nan
        }

    try:
        statistic, p_value = friedmanchisquare(*arrays)
    except ValueError:
        statistic, p_value = np.nan, np.nan

    return {
        "analysis": analysis_name,
        "patient_count": len(group),
        "friedman_statistic": statistic,
        "friedman_p_value": p_value
    }


def run_pairwise_tests(group, analysis_name):
    records = []

    for model_a, model_b in combinations(MODELS, 2):
        values_a = group[model_a].to_numpy(dtype=float)
        values_b = group[model_b].to_numpy(dtype=float)

        differences = values_b - values_a

        try:
            statistic, p_value = wilcoxon(
                values_a,
                values_b,
                alternative="two-sided",
                zero_method="wilcox",
                method="auto"
            )
        except ValueError:
            statistic, p_value = np.nan, np.nan

        mean_difference, ci_lower, ci_upper = (
            bootstrap_mean_difference(differences)
        )

        records.append({
            "analysis": analysis_name,
            "comparison": f"{model_b} vs {model_a}",
            "model_a": model_a,
            "model_b": model_b,
            "patient_count": len(group),
            "mean_dice_model_a": np.mean(values_a),
            "mean_dice_model_b": np.mean(values_b),
            "mean_dice_difference": mean_difference,
            "median_dice_difference": np.median(differences),
            "bootstrap_95_ci_lower": ci_lower,
            "bootstrap_95_ci_upper": ci_upper,
            "wilcoxon_statistic": statistic,
            "p_value": p_value
        })

    adjusted = holm_adjust([
        record["p_value"] for record in records
    ])

    for record, adjusted_p in zip(records, adjusted):
        record["holm_adjusted_p_value"] = adjusted_p
        record["significant_after_holm"] = (
            adjusted_p < 0.05
            if np.isfinite(adjusted_p)
            else False
        )

    return records


def run_analysis(data):
    friedman_records = []
    pairwise_records = []

    analyses = [("overall", data)]

    for class_name in CLASSES:
        class_data = data[
            data["label"] == class_name
        ].copy()

        analyses.append((class_name, class_data))

    for analysis_name, group in analyses:
        print(f"\nANALYSIS: {analysis_name.upper()}")
        print("-" * 70)
        print(f"Patients: {len(group)}")

        friedman_records.append(
            run_friedman(group, analysis_name)
        )

        pairwise_records.extend(
            run_pairwise_tests(group, analysis_name)
        )

    return (
        pd.DataFrame(friedman_records),
        pd.DataFrame(pairwise_records)
    )


def create_model_summary(data):
    records = []

    for model in MODELS:
        records.append({
            "model": model,
            "patient_count": len(data),
            "mean_patient_dice": data[model].mean(),
            "std_patient_dice": data[model].std(),
            "median_patient_dice": data[model].median(),
            "minimum_patient_dice": data[model].min(),
            "maximum_patient_dice": data[model].max()
        })

    return pd.DataFrame(records)


def print_results(model_summary, friedman, pairwise):
    print("\nPATIENT-LEVEL MODEL SUMMARY")
    print("=" * 100)
    print(model_summary.to_string(index=False))

    print("\nFRIEDMAN TEST RESULTS")
    print("=" * 100)
    print(friedman.to_string(index=False))

    print("\nPAIRWISE WILCOXON RESULTS")
    print("=" * 120)

    columns = [
        "analysis",
        "comparison",
        "patient_count",
        "mean_dice_difference",
        "bootstrap_95_ci_lower",
        "bootstrap_95_ci_upper",
        "p_value",
        "holm_adjusted_p_value",
        "significant_after_holm"
    ]

    print(
        pairwise[columns].to_string(
            index=False,
            float_format=lambda value: f"{value:.6f}"
        )
    )


def save_results(data, model_summary, friedman, pairwise):
    data.to_csv(
        OUTPUT_DIR / "patient_level_wide_results.csv",
        index=False
    )

    model_summary.to_csv(
        OUTPUT_DIR / "patient_level_model_summary.csv",
        index=False
    )

    friedman.to_csv(
        OUTPUT_DIR / "friedman_test_results.csv",
        index=False
    )

    pairwise.to_csv(
        OUTPUT_DIR / "pairwise_wilcoxon_results.csv",
        index=False
    )

    print("\nSAVED RESULTS")
    print("=" * 70)

    for filename in [
        "patient_level_wide_results.csv",
        "patient_level_model_summary.csv",
        "friedman_test_results.csv",
        "pairwise_wilcoxon_results.csv"
    ]:
        print(OUTPUT_DIR / filename)


def main():
    np.random.seed(RANDOM_SEED)
    create_directories()

    print("\nTHREE-MODEL PATIENT-LEVEL STATISTICAL ANALYSIS")
    print("=" * 75)

    data = load_patient_results()

    model_summary = create_model_summary(data)

    friedman, pairwise = run_analysis(data)

    print_results(
        model_summary,
        friedman,
        pairwise
    )

    save_results(
        data,
        model_summary,
        friedman,
        pairwise
    )

    print("\nSTATISTICAL ANALYSIS COMPLETED")


if __name__ == "__main__":
    main()

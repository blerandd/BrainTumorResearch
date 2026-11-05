
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_PATH = (
    PROJECT_ROOT
    / "results"
    / "segmentation"
    / "baseline_vs_augmented"
    / "patient_comparison.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "segmentation"
    / "baseline_vs_augmented"
    / "statistical_analysis"
)

BOOTSTRAP_ITERATIONS = 10000
SEED = 42


def load_patient_results():
    if not INPUT_PATH.is_file():
        raise FileNotFoundError(
            f"Patient comparison file not found: {INPUT_PATH}\n"
            "Run compare_baseline_augmented.py first."
        )

    data = pd.read_csv(INPUT_PATH)

    required = {
        "model",
        "patient_id",
        "label",
        "dice_mean"
    }

    missing = required - set(data.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    if data.duplicated(["model", "patient_id"]).any():
        raise ValueError(
            "Duplicate patient/model records found."
        )

    baseline = (
        data[data["model"] == "baseline"]
        .set_index("patient_id")
    )

    augmented = (
        data[data["model"] == "augmented"]
        .set_index("patient_id")
    )

    common_patients = baseline.index.intersection(
        augmented.index
    )

    if len(common_patients) != len(baseline) or len(common_patients) != len(augmented):
        raise ValueError(
            "Baseline and augmented models do not contain "
            "the same patients."
        )

    baseline = baseline.loc[common_patients].sort_index()
    augmented = augmented.loc[common_patients].sort_index()

    if not baseline["label"].equals(augmented["label"]):
        raise ValueError(
            "Patient class labels do not match between models."
        )

    paired = pd.DataFrame({
        "patient_id": common_patients,
        "label": baseline["label"].values,
        "baseline_dice": baseline["dice_mean"].values,
        "augmented_dice": augmented["dice_mean"].values
    })

    paired["dice_change"] = (
        paired["augmented_dice"] - paired["baseline_dice"]
    )

    return paired


def bootstrap_mean_ci(differences):
    rng = np.random.default_rng(SEED)

    differences = np.asarray(differences, dtype=float)

    bootstrap_means = np.empty(BOOTSTRAP_ITERATIONS)

    for index in range(BOOTSTRAP_ITERATIONS):
        sample = rng.choice(
            differences,
            size=len(differences),
            replace=True
        )

        bootstrap_means[index] = np.mean(sample)

    lower, upper = np.percentile(
        bootstrap_means,
        [2.5, 97.5]
    )

    return lower, upper


def analyze_primary_metric(paired):
    differences = paired["dice_change"].to_numpy()

    mean_change = np.mean(differences)
    median_change = np.median(differences)
    std_change = np.std(differences, ddof=1)

    ci_lower, ci_upper = bootstrap_mean_ci(differences)

    test = wilcoxon(
        paired["augmented_dice"],
        paired["baseline_dice"],
        alternative="two-sided",
        zero_method="wilcox",
        method="auto"
    )

    improved = int(np.sum(differences > 0))
    deteriorated = int(np.sum(differences < 0))
    unchanged = int(np.sum(differences == 0))

    summary = {
        "patient_count": len(paired),
        "baseline_mean_dice": paired["baseline_dice"].mean(),
        "augmented_mean_dice": paired["augmented_dice"].mean(),
        "mean_dice_change": mean_change,
        "median_dice_change": median_change,
        "std_dice_change": std_change,
        "bootstrap_95_ci_lower": ci_lower,
        "bootstrap_95_ci_upper": ci_upper,
        "wilcoxon_statistic": float(test.statistic),
        "wilcoxon_p_value": float(test.pvalue),
        "patients_improved": improved,
        "patients_deteriorated": deteriorated,
        "patients_unchanged": unchanged,
        "improved_percentage": improved / len(paired) * 100,
        "deteriorated_percentage": deteriorated / len(paired) * 100
    }

    return pd.DataFrame([summary])


def analyze_by_class(paired):
    rows = []

    for label, group in paired.groupby("label"):
        differences = group["dice_change"]

        rows.append({
            "label": label,
            "patient_count": len(group),
            "baseline_mean_dice": group["baseline_dice"].mean(),
            "augmented_mean_dice": group["augmented_dice"].mean(),
            "mean_dice_change": differences.mean(),
            "median_dice_change": differences.median(),
            "patients_improved": int((differences > 0).sum()),
            "patients_deteriorated": int((differences < 0).sum()),
            "patients_unchanged": int((differences == 0).sum())
        })

    return pd.DataFrame(rows)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("\nPATIENT-LEVEL STATISTICAL COMPARISON")
    print("=" * 75)

    paired = load_patient_results()

    print(f"Patients analyzed: {len(paired)}")
    print("\nPRIMARY ANALYSIS: PATIENT-LEVEL DICE")

    summary = analyze_primary_metric(paired)
    class_summary = analyze_by_class(paired)

    print("\nOVERALL STATISTICAL RESULTS")
    print("=" * 75)
    print(summary.T.to_string(header=False))

    print("\nRESULTS BY TUMOR CLASS")
    print("=" * 75)
    print(class_summary.to_string(index=False))

    paired.to_csv(
        OUTPUT_DIR / "patient_level_paired_results.csv",
        index=False
    )

    summary.to_csv(
        OUTPUT_DIR / "patient_level_statistical_summary.csv",
        index=False
    )

    class_summary.to_csv(
        OUTPUT_DIR / "patient_level_class_summary.csv",
        index=False
    )

    print("\nSAVED FILES")
    print("=" * 75)
    print(OUTPUT_DIR / "patient_level_paired_results.csv")
    print(OUTPUT_DIR / "patient_level_statistical_summary.csv")
    print(OUTPUT_DIR / "patient_level_class_summary.csv")

    print("\nPATIENT-LEVEL STATISTICAL ANALYSIS COMPLETED")


if __name__ == "__main__":
    main()

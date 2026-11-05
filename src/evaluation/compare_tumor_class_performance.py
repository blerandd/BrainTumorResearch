
from pathlib import Path
from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import kruskal, mannwhitneyu


BASE_DIR = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    BASE_DIR
    / "results"
    / "segmentation"
    / "final_evaluation"
    / "final_metrics_by_patient.csv"
)

OUTPUT_DIR = (
    BASE_DIR
    / "results"
    / "segmentation"
    / "statistical_analysis"
    / "tumor_class_comparison"
)

CLASSES = ["glioma", "meningioma", "pituitary"]


def holm_correction(p_values):
    p_values = np.asarray(p_values, dtype=float)
    count = len(p_values)

    order = np.argsort(p_values)
    adjusted = np.empty(count, dtype=float)

    running_max = 0.0

    for rank, index in enumerate(order):
        corrected = (count - rank) * p_values[index]
        running_max = max(running_max, corrected)
        adjusted[index] = min(running_max, 1.0)

    return adjusted


def cliffs_delta(group_a, group_b):
    greater = 0
    lower = 0

    for value_a in group_a:
        greater += np.sum(value_a > group_b)
        lower += np.sum(value_a < group_b)

    total = len(group_a) * len(group_b)

    if total == 0:
        return np.nan

    return (greater - lower) / total


def describe_groups(data):
    rows = []

    for label in CLASSES:
        values = data.loc[
            data["label"] == label, "dice_mean"
        ].dropna().to_numpy(dtype=float)

        rows.append({
            "label": label,
            "patient_count": len(values),
            "mean_dice": np.mean(values) if len(values) else np.nan,
            "std_dice": np.std(values, ddof=1) if len(values) > 1 else np.nan,
            "median_dice": np.median(values) if len(values) else np.nan,
            "q1_dice": np.percentile(values, 25) if len(values) else np.nan,
            "q3_dice": np.percentile(values, 75) if len(values) else np.nan,
            "min_dice": np.min(values) if len(values) else np.nan,
            "max_dice": np.max(values) if len(values) else np.nan
        })

    return pd.DataFrame(rows)


def compare_three_classes(data):
    groups = {
        label: data.loc[
            data["label"] == label, "dice_mean"
        ].dropna().to_numpy(dtype=float)
        for label in CLASSES
    }

    if any(len(values) == 0 for values in groups.values()):
        raise ValueError("At least one tumor class has no valid Dice values.")

    statistic, p_value = kruskal(
        *[groups[label] for label in CLASSES]
    )

    total_count = sum(len(values) for values in groups.values())
    group_count = len(groups)

    epsilon_squared = (
        statistic - group_count + 1
    ) / (total_count - group_count)

    epsilon_squared = max(0.0, epsilon_squared)

    result = pd.DataFrame([{
        "test": "Kruskal-Wallis",
        "statistic": statistic,
        "p_value": p_value,
        "epsilon_squared": epsilon_squared,
        "patient_count": total_count,
        "group_count": group_count
    }])

    return result, groups


def pairwise_comparisons(groups):
    rows = []

    for label_a, label_b in combinations(CLASSES, 2):
        values_a = groups[label_a]
        values_b = groups[label_b]

        statistic, p_value = mannwhitneyu(
            values_a,
            values_b,
            alternative="two-sided"
        )

        delta = cliffs_delta(values_a, values_b)

        rows.append({
            "comparison": f"{label_a} vs {label_b}",
            "group_a": label_a,
            "group_b": label_b,
            "n_a": len(values_a),
            "n_b": len(values_b),
            "median_a": np.median(values_a),
            "median_b": np.median(values_b),
            "mann_whitney_u": statistic,
            "p_value": p_value,
            "cliffs_delta": delta
        })

    results = pd.DataFrame(rows)

    results["p_value_holm"] = holm_correction(
        results["p_value"].to_numpy()
    )

    results["significant_0_05"] = (
        results["p_value_holm"] < 0.05
    )

    return results


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading patient-level final test results...")

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input file not found: {INPUT_FILE}"
        )

    data = pd.read_csv(INPUT_FILE)

    required_columns = {"patient_id", "label", "dice_mean"}

    missing = required_columns - set(data.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    data = data[data["label"].isin(CLASSES)].copy()
    data["dice_mean"] = pd.to_numeric(
        data["dice_mean"], errors="coerce"
    )

    data = data.dropna(subset=["patient_id", "label", "dice_mean"])

    if not data["dice_mean"].between(0, 1).all():
        raise ValueError("Dice values must be between 0 and 1.")

    if data["patient_id"].duplicated().any():
        raise ValueError(
            "Duplicate patient IDs detected. "
            "Expected one row per patient."
        )

    print(f"Valid patients: {len(data)}")

    print("\nDESCRIPTIVE STATISTICS")
    descriptive = describe_groups(data)
    print(descriptive.to_string(index=False))

    print("\nKRUSKAL-WALLIS TEST")

    kruskal_results, groups = compare_three_classes(data)
    print(kruskal_results.to_string(index=False))

    print("\nPAIRWISE MANN-WHITNEY U TESTS")

    pairwise_results = pairwise_comparisons(groups)
    print(pairwise_results.to_string(index=False))

    descriptive.to_csv(
        OUTPUT_DIR / "class_dice_descriptive_statistics.csv",
        index=False
    )

    kruskal_results.to_csv(
        OUTPUT_DIR / "kruskal_wallis_results.csv",
        index=False
    )

    pairwise_results.to_csv(
        OUTPUT_DIR / "pairwise_class_comparisons.csv",
        index=False
    )

    print("\nINTERPRETATION")

    p_value = kruskal_results.iloc[0]["p_value"]

    if p_value < 0.05:
        print(
            "The Kruskal-Wallis test indicates a statistically "
            "significant difference among the tumor classes."
        )
    else:
        print(
            "The Kruskal-Wallis test does not provide sufficient "
            "evidence of a difference among the tumor classes."
        )

    print("\nPairwise comparisons after Holm correction:")

    for _, row in pairwise_results.iterrows():
        status = (
            "SIGNIFICANT"
            if row["significant_0_05"]
            else "NOT SIGNIFICANT"
        )

        print(
            f"{row['comparison']}: "
            f"adjusted p={row['p_value_holm']:.6f} "
            f"({status})"
        )

    print("\nStatistical comparison completed successfully.")
    print(f"Results saved in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

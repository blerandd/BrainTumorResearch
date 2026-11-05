
from pathlib import Path

import pandas as pd
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests


INPUT = Path(
    "results/segmentation/failure_analysis/"
    "paired_failure_comparison.csv"
)

OUTPUT_DIR = Path(
    "results/segmentation/statistical_analysis/"
    "patient_error_changes"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(INPUT)

df["fp_change"] = df["augmented_fp"] - df["baseline_fp"]
df["fn_change"] = df["augmented_fn"] - df["baseline_fn"]
df["total_error_change"] = df["fp_change"] + df["fn_change"]

patient_df = (
    df.groupby(["patient_id", "label"])
    .agg(
        fp_change=("fp_change", "mean"),
        fn_change=("fn_change", "mean"),
        total_error_change=("total_error_change", "mean"),
        image_count=("sample_id", "count")
    )
    .reset_index()
)

patient_df.to_csv(
    OUTPUT_DIR / "patient_error_changes.csv",
    index=False
)

results = []

analyses = {
    "overall": patient_df
}

for label, group in patient_df.groupby("label"):
    analyses[label] = group

metrics = [
    "fp_change",
    "fn_change",
    "total_error_change"
]

for analysis, group in analyses.items():
    for metric in metrics:
        values = group[metric].dropna()

        if len(values) < 2 or (values == 0).all():
            continue

        statistic, p_value = wilcoxon(
            values,
            alternative="two-sided"
        )

        results.append({
            "analysis": analysis,
            "metric": metric,
            "patient_count": len(values),
            "mean_change": values.mean(),
            "median_change": values.median(),
            "wilcoxon_statistic": statistic,
            "p_value": p_value
        })

results_df = pd.DataFrame(results)

results_df["holm_adjusted_p"] = multipletests(
    results_df["p_value"],
    method="holm"
)[1]

results_df["significant_after_holm"] = (
    results_df["holm_adjusted_p"] < 0.05
)

results_df.to_csv(
    OUTPUT_DIR / "patient_error_change_statistics.csv",
    index=False
)

print("\nPATIENT ERROR CHANGE SUMMARY")
print(patient_df.groupby("label")[
    ["fp_change", "fn_change", "total_error_change"]
].mean().round(2))

print("\nSTATISTICAL TEST RESULTS")
print(results_df.round(5).to_string(index=False))

print("\nSaved results to:", OUTPUT_DIR)

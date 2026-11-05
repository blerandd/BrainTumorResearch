
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from scipy.stats import bootstrap


INPUT = Path(
    "results/segmentation/statistical_analysis/"
    "patient_error_changes/patient_error_changes.csv"
)

OUTPUT = INPUT.parent / "error_effect_sizes.csv"

df = pd.read_csv(INPUT)

metrics = [
    "fp_change",
    "fn_change",
    "total_error_change"
]

groups = {"overall": df}

for label, group in df.groupby("label"):
    groups[label] = group

results = []

for group_name, group in groups.items():
    for metric in metrics:
        values = group[metric].dropna().to_numpy()

        if len(values) < 2:
            continue

        ci = bootstrap(
            (values,),
            np.mean,
            confidence_level=0.95,
            n_resamples=10000,
            method="BCa",
            random_state=42
        )

        nonzero = values[values != 0]

        if len(nonzero):
            hl_estimate = np.median(
                [
                    (values[i] + values[j]) / 2
                    for i in range(len(values))
                    for j in range(i, len(values))
                ]
            )
        else:
            hl_estimate = 0.0

        results.append({
            "group": group_name,
            "metric": metric,
            "patient_count": len(values),
            "mean_change": np.mean(values),
            "median_change": np.median(values),
            "mean_ci_lower": ci.confidence_interval.low,
            "mean_ci_upper": ci.confidence_interval.high,
            "patients_reduced": int((values < 0).sum()),
            "patients_increased": int((values > 0).sum()),
            "patients_unchanged": int((values == 0).sum()),
            "hodges_lehmann_estimate": hl_estimate
        })

results_df = pd.DataFrame(results)

results_df.to_csv(OUTPUT, index=False)

print("\nERROR CHANGE EFFECT SIZES")
print(results_df.round(4).to_string(index=False))

print("\nSaved:", OUTPUT)


from pathlib import Path

import pandas as pd
from scipy.stats import spearmanr


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_PATH = (
    PROJECT_ROOT
    / "results/segmentation/statistical_analysis/augmentation_morphology/"
    / "glioma_augmentation_morphology_merged.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results/segmentation/statistical_analysis/augmentation_morphology"
)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    data = pd.read_csv(INPUT_PATH)

    full_rho, full_p = spearmanr(
        data["compactness"],
        data["dice_improvement"]
    )

    results = []

    for index, row in data.iterrows():
        reduced = data.drop(index)

        rho, p_value = spearmanr(
            reduced["compactness"],
            reduced["dice_improvement"]
        )

        results.append({
            "removed_patient": row["patient_id"],
            "remaining_patients": len(reduced),
            "spearman_rho": rho,
            "p_value": p_value
        })

    results_df = pd.DataFrame(results)

    print("\nCOMPACTNESS SENSITIVITY ANALYSIS")
    print("=" * 65)
    print(f"All patients: {len(data)}")
    print(f"Original Spearman rho: {full_rho:.4f}")
    print(f"Original p-value: {full_p:.4f}")

    print("\nLEAVE-ONE-PATIENT-OUT RESULTS")
    print("=" * 65)
    print(results_df.to_string(index=False))

    print("\nSENSITIVITY SUMMARY")
    print("=" * 65)
    print(f"Minimum rho: {results_df['spearman_rho'].min():.4f}")
    print(f"Maximum rho: {results_df['spearman_rho'].max():.4f}")
    print(
        "Negative correlations: "
        f"{(results_df['spearman_rho'] < 0).sum()}"
    )
    print(
        "Positive correlations: "
        f"{(results_df['spearman_rho'] > 0).sum()}"
    )

    results_df.to_csv(
        OUTPUT_DIR / "compactness_leave_one_out.csv",
        index=False
    )

    print("\nSaved:")
    print(OUTPUT_DIR / "compactness_leave_one_out.csv")


if __name__ == "__main__":
    main()

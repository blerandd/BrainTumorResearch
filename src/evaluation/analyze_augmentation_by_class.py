
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_PATH = (
    PROJECT_ROOT
    / "results/segmentation/statistical_comparison/patient_dice_differences.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results/segmentation/statistical_analysis/augmentation_morphology"
)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    data = pd.read_csv(INPUT_PATH)

    summary = (
        data.groupby("label")
        .agg(
            patient_count=("patient_id", "count"),
            improved=("dice_improvement", lambda x: (x > 0).sum()),
            declined=("dice_improvement", lambda x: (x < 0).sum()),
            mean_improvement=("dice_improvement", "mean"),
            median_improvement=("dice_improvement", "median"),
            std_improvement=("dice_improvement", "std"),
            minimum_improvement=("dice_improvement", "min"),
            maximum_improvement=("dice_improvement", "max")
        )
        .reset_index()
    )

    summary["improved_percentage"] = (
        summary["improved"] / summary["patient_count"] * 100
    )

    print("\nAUGMENTATION EFFECT BY TUMOR CLASS")
    print("=" * 75)
    print(summary.to_string(index=False))

    print("\nINTERPRETATION")
    print("=" * 75)

    for _, row in summary.iterrows():
        print(
            f"{row['label']}: "
            f"{row['improved']}/{row['patient_count']} patients improved "
            f"({row['improved_percentage']:.1f}%). "
            f"Mean Dice change: {row['mean_improvement']:+.4f}; "
            f"median: {row['median_improvement']:+.4f}."
        )

    summary.to_csv(
        OUTPUT_DIR / "augmentation_effect_by_class.csv",
        index=False
    )

    print("\nSaved:")
    print(OUTPUT_DIR / "augmentation_effect_by_class.csv")


if __name__ == "__main__":
    main()

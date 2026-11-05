
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

    data = data.sort_values(
        "dice_improvement",
        ascending=False
    )

    print("\nAUGMENTATION PATIENT EXTREMES")
    print("=" * 75)

    print("\nTOP 10 PATIENTS WITH THE LARGEST IMPROVEMENT")
    print("=" * 75)

    print(
        data.head(10).to_string(index=False)
    )

    print("\nTOP 10 PATIENTS WITH THE LARGEST DECREASE")
    print("=" * 75)

    print(
        data.tail(10)
        .sort_values("dice_improvement")
        .to_string(index=False)
    )

    print("\nIMPROVEMENT DISTRIBUTION")
    print("=" * 75)

    print(f"Total patients: {len(data)}")
    print(
        f"Improved: {(data['dice_improvement'] > 0).sum()}"
    )
    print(
        f"Unchanged: {(data['dice_improvement'] == 0).sum()}"
    )
    print(
        f"Declined: {(data['dice_improvement'] < 0).sum()}"
    )

    print(
        f"Mean improvement: {data['dice_improvement'].mean():.4f}"
    )
    print(
        f"Median improvement: {data['dice_improvement'].median():.4f}"
    )
    print(
        f"Minimum improvement: {data['dice_improvement'].min():.4f}"
    )
    print(
        f"Maximum improvement: {data['dice_improvement'].max():.4f}"
    )

    data.to_csv(
        OUTPUT_DIR / "patient_augmentation_rankings.csv",
        index=False
    )

    print("\nSaved:")
    print(OUTPUT_DIR / "patient_augmentation_rankings.csv")


if __name__ == "__main__":
    main()

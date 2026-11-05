
from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[2]

CSV_PATH = (
    BASE_DIR
    / "results"
    / "segmentation"
    / "boundary_metrics"
    / "boundary_metrics_by_image.csv"
)


def main():
    df = pd.read_csv(CSV_PATH)

    infinite_cases = df[
        df["hd95_pixels"].isin([float("inf")])
        | df["assd_pixels"].isin([float("inf")])
    ].copy()

    print("TOTAL TEST IMAGES:", len(df))
    print("INFINITE DISTANCE CASES:", len(infinite_cases))

    print("\nEMPTY MASK COMBINATIONS")

    print(
        pd.crosstab(
            infinite_cases["ground_truth_empty"],
            infinite_cases["prediction_empty"],
            rownames=["Ground truth empty"],
            colnames=["Prediction empty"],
            margins=True
        )
    )

    print("\nINFINITE CASES BY CLASS")

    print(
        infinite_cases.groupby("label").size()
    )

    print("\nDETAILED INFINITE CASES")

    columns = [
        "sample_id",
        "patient_id",
        "label",
        "ground_truth_empty",
        "prediction_empty",
        "hd95_pixels",
        "assd_pixels",
        "boundary_dice"
    ]

    print(
        infinite_cases[columns].to_string(index=False)
    )

    output_path = (
        BASE_DIR
        / "results"
        / "segmentation"
        / "boundary_metrics"
        / "infinite_distance_cases.csv"
    )

    infinite_cases[columns].to_csv(output_path, index=False)

    print("\nSaved:", output_path)


if __name__ == "__main__":
    main()

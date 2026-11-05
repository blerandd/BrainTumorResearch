
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


BASE_DIR = Path(__file__).resolve().parents[2]

INPUT_PATH = (
    BASE_DIR
    / "results"
    / "segmentation"
    / "training_diagnostics"
    / "partition_distribution.csv"
)

OUTPUT_DIR = (
    BASE_DIR
    / "results"
    / "segmentation"
    / "training_diagnostics"
)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(INPUT_PATH)

    patient_data = (
        df.groupby(
            ["split", "label", "patient_id"]
        )
        .agg(
            image_count=("sample_id", "count"),
            mean_tumor_area=("tumor_area_percent", "mean"),
            median_tumor_area=("tumor_area_percent", "median"),
            minimum_tumor_area=("tumor_area_percent", "min"),
            maximum_tumor_area=("tumor_area_percent", "max")
        )
        .reset_index()
    )

    patient_data.to_csv(
        OUTPUT_DIR / "patient_tumor_distribution.csv",
        index=False
    )

    print("\nPATIENT-LEVEL TUMOR AREA SUMMARY")

    summary = (
        patient_data
        .groupby(["split", "label"])
        .agg(
            patient_count=("patient_id", "nunique"),
            median_patient_area=("median_tumor_area", "median"),
            mean_patient_area=("mean_tumor_area", "mean"),
            minimum_patient_area=("median_tumor_area", "min"),
            maximum_patient_area=("median_tumor_area", "max")
        )
    )

    print(summary.to_string())

    print("\nPITUITARY PATIENT DETAILS")

    pituitary = patient_data[
        patient_data["label"] == "pituitary"
    ].sort_values(
        ["split", "median_tumor_area"]
    )

    print(
        pituitary.to_string(index=False)
    )

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(16, 5)
    )

    classes = ["glioma", "meningioma", "pituitary"]
    splits = ["train", "validation", "test"]

    for ax, class_name in zip(axes, classes):
        plot_data = [
            patient_data[
                (patient_data["label"] == class_name)
                & (patient_data["split"] == split)
            ]["median_tumor_area"].values
            for split in splits
        ]

        ax.boxplot(
            plot_data,
            tick_labels=splits
        )

        ax.set_title(class_name.capitalize())
        ax.set_ylabel("Patient median tumor area (%)")
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle(
        "Patient-Level Tumor Area Distribution",
        fontsize=14
    )

    fig.tight_layout()

    fig.savefig(
        OUTPUT_DIR / "patient_tumor_area_distribution.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)

    print("\nSaved patient-level analysis:")
    print(OUTPUT_DIR / "patient_tumor_distribution.csv")
    print(OUTPUT_DIR / "patient_tumor_area_distribution.png")


if __name__ == "__main__":
    main()

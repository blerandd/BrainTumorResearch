
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

from src.data.dataset_loader import BrainTumorDatasetLoader


BASE_DIR = Path(__file__).resolve().parents[2]


def main():
    loader = BrainTumorDatasetLoader(
        image_size=(256, 256),
        batch_size=16
    )

    rows = []

    for split in ["train", "validation", "test"]:
        print(f"Analyzing {split} partition...")

        dataset = loader.get_segmentation_datasets()[split]

        image_index = 0

        for images, masks in dataset:
            image_array = images.numpy()
            mask_array = masks.numpy()

            for index in range(len(image_array)):
                image = image_array[index].squeeze()
                mask = mask_array[index].squeeze() > 0.5

                tumor_pixels = int(np.sum(mask))

                rows.append({
                    "sample_id": None,
                    "split": split,
                    "tumor_pixels": tumor_pixels,
                    "tumor_area_percent": (
                        tumor_pixels / mask.size * 100
                    ),
                    "image_mean": float(np.mean(image)),
                    "image_std": float(np.std(image)),
                    "image_min": float(np.min(image)),
                    "image_max": float(np.max(image))
                })

                image_index += 1

        print(f"{split}: {image_index} images")

    results = pd.DataFrame(rows)

    metadata = loader.metadata.copy()

    for split in ["train", "validation", "test"]:
        split_indices = results.index[results["split"] == split]
        split_metadata = metadata[
            metadata["split"] == split
        ].reset_index(drop=True)

        if len(split_indices) != len(split_metadata):
            raise ValueError(
                f"Metadata mismatch in {split} partition."
            )

        results.loc[split_indices, "sample_id"] = (
            split_metadata["sample_id"].astype(str).to_numpy()
        )

        results.loc[split_indices, "patient_id"] = (
            split_metadata["patient_id"].astype(str).to_numpy()
        )

        results.loc[split_indices, "label"] = (
            split_metadata["label"].to_numpy()
        )

    output_dir = (
        BASE_DIR
        / "results"
        / "segmentation"
        / "training_diagnostics"
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    results.to_csv(
        output_dir / "partition_distribution.csv",
        index=False
    )

    print("\nPARTITION SUMMARY")

    print(
        results.groupby("split").agg(
            image_count=("sample_id", "count"),
            patient_count=("patient_id", "nunique"),
            mean_tumor_area=("tumor_area_percent", "mean"),
            median_tumor_area=("tumor_area_percent", "median"),
            minimum_tumor_area=("tumor_area_percent", "min"),
            maximum_tumor_area=("tumor_area_percent", "max"),
            mean_image_intensity=("image_mean", "mean"),
            mean_image_std=("image_std", "mean")
        ).to_string()
    )

    print("\nCLASS DISTRIBUTION")

    print(
        results.groupby(["split", "label"]).agg(
            images=("sample_id", "count"),
            mean_tumor_area=("tumor_area_percent", "mean"),
            median_tumor_area=("tumor_area_percent", "median")
        ).to_string()
    )

    print("\nSaved partition distribution:")
    print(output_dir / "partition_distribution.csv")


if __name__ == "__main__":
    main()

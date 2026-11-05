
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "processed"

METADATA_PATH = DATA_DIR / "metadata.csv"
SPLITS_PATH = DATA_DIR / "splits.csv"


def main():
    metadata = pd.read_csv(METADATA_PATH)
    splits = pd.read_csv(SPLITS_PATH)

    print("\nMETADATA COLUMNS")
    print("=" * 55)
    print(metadata.columns.tolist())

    print("\nFIRST METADATA RECORDS")
    print("=" * 55)
    print(metadata.head(3).to_string(index=False))

    print("\nSPLIT COLUMNS")
    print("=" * 55)
    print(splits.columns.tolist())

    print("\nFIRST SPLIT RECORDS")
    print("=" * 55)
    print(splits.head(3).to_string(index=False))

    print("\nDATASET COUNTS")
    print("=" * 55)

    print("MRI samples:", len(metadata))
    print("Unique patients:", metadata["patient_id"].nunique())

    print("\nCLASS DISTRIBUTION")
    print(metadata["label"].value_counts())

    print("\nCURRENT PARTITIONS")
    print(splits["split"].value_counts())

    print("\nIMAGE AND MASK INSPECTION")
    print("=" * 55)

    successful_images = 0
    successful_masks = 0

    for _, row in metadata.head(5).iterrows():
        image_path = DATA_DIR / row["image_path"]
        mask_path = DATA_DIR / row["mask_path"]

        print("\nSample:", row["sample_id"])
        print("Patient:", row["patient_id"])
        print("Label:", row["label"])

        image_exists = image_path.is_file()
        mask_exists = mask_path.is_file()

        print("Image path:", image_path)
        print("Image exists:", image_exists)

        print("Mask path:", mask_path)
        print("Mask exists:", mask_exists)

        if image_exists:
            image = Image.open(image_path)
            image_array = np.asarray(image)

            print("\nMRI INFORMATION")
            print("Dimensions:", image.size)
            print("Mode:", image.mode)
            print("Data type:", image_array.dtype)
            print("Minimum intensity:", image_array.min())
            print("Maximum intensity:", image_array.max())

            successful_images += 1

        if mask_exists:
            mask = Image.open(mask_path)
            mask_array = np.asarray(mask)

            print("\nMASK INFORMATION")
            print("Dimensions:", mask.size)
            print("Mode:", mask.mode)
            print("Data type:", mask_array.dtype)
            print("Unique values:", np.unique(mask_array)[:20])

            successful_masks += 1

    print("\nINSPECTION SUMMARY")
    print("=" * 55)
    print("Images successfully inspected:", successful_images)
    print("Masks successfully inspected:", successful_masks)

    if successful_images == 5 and successful_masks == 5:
        print("RESULT: All inspected files were found.")
    else:
        print("WARNING: Some files could not be located.")

    print("\nINSPECTION COMPLETED")


if __name__ == "__main__":
    main()

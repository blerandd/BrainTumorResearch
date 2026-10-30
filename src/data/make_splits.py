from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = ROOT / "data" / "processed"
METADATA_PATH = PROCESSED_DIR / "metadata.csv"
SPLIT_PATH = PROCESSED_DIR / "splits.csv"
SEED = 42

def main():
    metadata = pd.read_csv(METADATA_PATH)
    if metadata.empty:
        raise ValueError("Metadata is empty.")

    patient_labels = metadata.groupby("patient_id")["label"].agg(
        lambda values: values.mode().iloc[0] if not values.mode().empty else values.iloc[0]
    )
    consistency = metadata.groupby("patient_id")["label"].nunique()
    if (consistency > 1).any():
        patients = consistency[consistency > 1].index.tolist()
        raise ValueError(f"Some patient IDs have multiple class labels: {patients[:10]}")

    patients = patient_labels.rename("label").reset_index()
    train_patients, temporary_patients = train_test_split(
        patients,
        test_size=0.30,
        random_state=SEED,
        stratify=patients["label"]
    )
    validation_patients, test_patients = train_test_split(
        temporary_patients,
        test_size=0.50,
        random_state=SEED,
        stratify=temporary_patients["label"]
    )

    split_map = {}
    split_map.update(dict.fromkeys(train_patients["patient_id"], "train"))
    split_map.update(dict.fromkeys(validation_patients["patient_id"], "validation"))
    split_map.update(dict.fromkeys(test_patients["patient_id"], "test"))

    metadata["split"] = metadata["patient_id"].map(split_map)
    if metadata["split"].isna().any():
        raise RuntimeError("Some samples were not assigned to a split.")

    metadata.to_csv(SPLIT_PATH, index=False)

    print("Patient-independent split summary")
    print(metadata.groupby("split").agg(
        images=("sample_id", "count"),
        patients=("patient_id", "nunique")
    ))
    print("\nClass distribution by split (image counts)")
    print(pd.crosstab(metadata["split"], metadata["label"]))
    print(f"\nSaved: {SPLIT_PATH}")

if __name__ == "__main__":
    main()

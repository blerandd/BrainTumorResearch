
from pathlib import Path

import mat73
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "processed"
CVIND_PATH = ROOT / "data" / "raw" / "cvind.mat"
METADATA_PATH = DATA_DIR / "metadata.csv"
SPLITS_PATH = DATA_DIR / "splits.csv"


def load_data():
    metadata = pd.read_csv(METADATA_PATH)
    splits = pd.read_csv(SPLITS_PATH)

    mat_data = mat73.loadmat(str(CVIND_PATH))
    cvind = np.asarray(mat_data["cvind"]).flatten()

    return metadata, splits, cvind


def validate_sample_identifiers(metadata, cvind):
    print("\nVERIFYING SAMPLE IDENTIFIERS")
    print("=" * 50)

    if "sample_id" not in metadata.columns:
        raise ValueError("metadata.csv does not contain sample_id.")

    sample_numbers = pd.to_numeric(
        metadata["sample_id"],
        errors="coerce"
    )

    if sample_numbers.isna().any():
        raise ValueError("Some sample IDs are not numeric.")

    if sample_numbers.duplicated().any():
        raise ValueError("Duplicate sample IDs detected.")

    expected_ids = set(range(1, len(cvind) + 1))
    actual_ids = set(sample_numbers.astype(int))

    if actual_ids != expected_ids:
        raise ValueError(
            "Sample IDs do not match the expected 1-based cvind indexing."
        )

    if len(metadata) != len(cvind):
        raise ValueError("Metadata and cvind have different lengths.")

    print("Metadata samples:", len(metadata))
    print("Cross-validation assignments:", len(cvind))
    print("Minimum sample ID:", sample_numbers.min())
    print("Maximum sample ID:", sample_numbers.max())
    print("Sample identifiers verified.")

    return sample_numbers.astype(int)


def analyze_original_folds(metadata, cvind, sample_numbers):
    print("\nORIGINAL CROSS-VALIDATION FOLDS")
    print("=" * 50)

    df = metadata.copy()

    df["sample_number"] = sample_numbers

    fold_mapping = pd.Series(
        cvind,
        index=np.arange(1, len(cvind) + 1)
    )

    df["cvind_fold"] = df["sample_number"].map(fold_mapping)

    if df["cvind_fold"].isna().any():
        raise ValueError("Some samples have no fold assignment.")

    print("Total MRI samples:", len(df))
    print("Unique folds:", sorted(df["cvind_fold"].unique()))

    print("\nFOLD DISTRIBUTION")
    print(df["cvind_fold"].value_counts().sort_index())

    print("\nFOLD DISTRIBUTION BY CLASS")
    print(pd.crosstab(df["cvind_fold"], df["label"]))

    return df


def verify_patient_independence(df):
    print("\nPATIENT INDEPENDENCE AUDIT")
    print("=" * 50)

    patient_folds = df.groupby("patient_id")["cvind_fold"].nunique()

    multiple_folds = int((patient_folds > 1).sum())

    print("Unique patients:", len(patient_folds))
    print("Patients assigned to multiple folds:", multiple_folds)
    print("Maximum folds assigned to one patient:", patient_folds.max())

    if multiple_folds == 0:
        print("RESULT: All patients remain within a single fold.")
    else:
        print("WARNING: Patient leakage detected in original folds.")

        leaking_patients = patient_folds[patient_folds > 1].index.tolist()

        print("Patients appearing in multiple folds:", leaking_patients)


def compare_current_split(df, splits):
    print("\nCURRENT SPLIT VS ORIGINAL FOLDS")
    print("=" * 50)

    required_columns = {"patient_id", "split"}

    if not required_columns.issubset(splits.columns):
        raise ValueError(
            "splits.csv must contain patient_id and split columns."
        )

    patient_split_counts = splits.groupby("patient_id")["split"].nunique()

    leaking_patients = patient_split_counts[patient_split_counts > 1]

    if not leaking_patients.empty:
        raise ValueError(
            "Some patients appear in multiple current splits."
        )

    patient_splits = splits[
        ["patient_id", "split"]
    ].drop_duplicates()

    comparison = df.merge(
        patient_splits,
        on="patient_id",
        how="left",
        validate="many_to_one"
    )

    missing = int(comparison["split"].isna().sum())

    print("Missing current split assignments:", missing)

    if missing:
        raise ValueError(
            "Some patients do not have a current split assignment."
        )

    print("\nIMAGE COUNTS: CURRENT SPLIT VS ORIGINAL FOLDS")
    print(pd.crosstab(comparison["split"], comparison["cvind_fold"]))

    print("\nPATIENT COUNTS: CURRENT SPLIT VS ORIGINAL FOLDS")

    patient_comparison = comparison[
        ["patient_id", "split", "cvind_fold"]
    ].drop_duplicates()

    print(
        pd.crosstab(
            patient_comparison["split"],
            patient_comparison["cvind_fold"]
        )
    )

    print("\nCURRENT SPLIT PATIENT COUNTS")
    print(
        patient_comparison.groupby("split")["patient_id"].nunique()
    )

    print("\nCURRENT SPLIT IMAGE COUNTS")
    print(comparison.groupby("split").size())


def main():
    print("\nFIGSHARE CROSS-VALIDATION ANALYSIS")
    print("=" * 50)

    metadata, splits, cvind = load_data()

    sample_numbers = validate_sample_identifiers(
        metadata,
        cvind
    )

    df = analyze_original_folds(
        metadata,
        cvind,
        sample_numbers
    )

    verify_patient_independence(df)

    compare_current_split(
        df,
        splits
    )

    print("\nANALYSIS COMPLETED SUCCESSFULLY")


if __name__ == "__main__":
    main()

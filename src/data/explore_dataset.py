
from pathlib import Path
import pandas as pd
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "processed"
METADATA_PATH = DATA_DIR / "metadata.csv"

df = pd.read_csv(METADATA_PATH)

print("\nDATASET OVERVIEW")
print("=" * 45)
print("Total MRI samples:", len(df))
print("Unique patients:", df["patient_id"].nunique())
print("Unique classes:", df["label"].nunique())

print("\nCLASS DISTRIBUTION")
print("=" * 45)
print(df["label"].value_counts())
print("\nClass percentages:")
print((df["label"].value_counts(normalize=True) * 100).round(2))

print("\nPATIENT DISTRIBUTION")
print("=" * 45)
patient_counts = df.groupby("patient_id").size()
print(patient_counts.describe())

print("\nPATIENT LABEL CONSISTENCY")
print("=" * 45)
label_counts = df.groupby("patient_id")["label"].nunique()
print("Patients with multiple labels:", (label_counts > 1).sum())

print("\nFILE VALIDATION")
print("=" * 45)

missing_images = 0
missing_masks = 0
empty_masks = 0

for row in df.itertuples():
    image_path = DATA_DIR / row.image_path
    mask_path = DATA_DIR / row.mask_path

    if not image_path.exists():
        missing_images += 1

    if not mask_path.exists():
        missing_masks += 1
    else:
        mask = np.array(Image.open(mask_path))
        if not np.any(mask):
            empty_masks += 1

print("Missing images:", missing_images)
print("Missing masks:", missing_masks)
print("Empty masks:", empty_masks)

print("\nIMAGE DIMENSIONS")
print("=" * 45)

sample = df.iloc[0]
image = Image.open(DATA_DIR / sample["image_path"])
mask = Image.open(DATA_DIR / sample["mask_path"])

print("MRI dimensions:", image.size)
print("Mask dimensions:", mask.size)
print("MRI mode:", image.mode)
print("Mask mode:", mask.mode)

print("\nAUDIT COMPLETED")

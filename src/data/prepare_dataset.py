from pathlib import Path
import csv
import numpy as np
import mat73
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
IMAGE_DIR = PROCESSED_DIR / "images"
MASK_DIR = PROCESSED_DIR / "masks"
METADATA_PATH = PROCESSED_DIR / "metadata.csv"

LABEL_NAMES = {
    1: "meningioma",
    2: "glioma",
    3: "pituitary"
}

def find_field(value, candidates):
    if isinstance(value, dict):
        lowered = {str(key).lower(): key for key in value}
        for candidate in candidates:
            if candidate.lower() in lowered:
                return value[lowered[candidate.lower()]]
        for nested in value.values():
            result = find_field(nested, candidates)
            if result is not None:
                return result
    return None

def scalar(value):
    array = np.asarray(value).squeeze()
    return array.item() if array.size == 1 else array

def normalize_image(image):
    image = np.asarray(image).squeeze().astype(np.float32)
    low, high = np.percentile(image, [1, 99])
    if high <= low:
        low, high = float(image.min()), float(image.max())
    if high <= low:
        return np.zeros(image.shape, dtype=np.uint16)
    scaled = np.clip((image - low) / (high - low), 0, 1)
    return (scaled * 65535).astype(np.uint16)

def process_file(path, writer):
    try:
        data = mat73.loadmat(str(path))
    except Exception as error:
        print(f"Skipped {path.name}: {error}")
        return False

    image = find_field(data, ["image"])
    mask = find_field(data, ["tumorMask", "tumormask", "mask"])
    label = find_field(data, ["label"])
    patient_id = find_field(data, ["PID", "pid", "patient_id"])

    if image is None or mask is None or label is None or patient_id is None:
        print(f"Skipped {path.name}: required fields not found")
        return False

    image = np.asarray(image).squeeze()
    mask = np.asarray(mask).squeeze()
    label = int(float(scalar(label)))
    patient_id = str(scalar(patient_id))

    if label not in LABEL_NAMES or image.ndim != 2 or mask.ndim != 2:
        print(f"Skipped {path.name}: unexpected image, mask, or label")
        return False

    image_name = f"{path.stem}.png"
    mask_name = f"{path.stem}_mask.png"
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    MASK_DIR.mkdir(parents=True, exist_ok=True)

    Image.fromarray(normalize_image(image)).save(IMAGE_DIR / image_name)
    binary_mask = (mask > 0).astype(np.uint8) * 255
    Image.fromarray(binary_mask).save(MASK_DIR / mask_name)

    writer.writerow({
        "sample_id": path.stem,
        "patient_id": patient_id,
        "label_id": label,
        "label": LABEL_NAMES[label],
        "image_path": str(Path("images") / image_name),
        "mask_path": str(Path("masks") / mask_name)
    })
    return True

def main():
    mat_files = sorted(RAW_DIR.rglob("*.mat"))
    if not mat_files:
        raise FileNotFoundError("No .mat files found. Run python -m src.data.download_figshare first.")

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    processed = 0
    with METADATA_PATH.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=[
            "sample_id", "patient_id", "label_id", "label", "image_path", "mask_path"
        ])
        writer.writeheader()
        for path in mat_files:
            processed += int(process_file(path, writer))

    print(f"Processed {processed} samples.")
    print(f"Metadata: {METADATA_PATH}")

if __name__ == "__main__":
    main()

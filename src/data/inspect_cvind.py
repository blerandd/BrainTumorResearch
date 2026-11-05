
from pathlib import Path
import mat73
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
CVIND_PATH = ROOT / "data" / "raw" / "cvind.mat"

data = mat73.loadmat(str(CVIND_PATH))

print("\nCVIND MATLAB STRUCTURE")
print("=" * 45)

for key, value in data.items():
    print(f"\nKey: {key}")
    print("Type:", type(value))

    if isinstance(value, dict):
        for subkey, subvalue in value.items():
            print(
                " ",
                subkey,
                "Shape:", getattr(subvalue, "shape", None),
                "Type:", type(subvalue)
            )

            if isinstance(subvalue, np.ndarray):
                print("  First values:", subvalue.flatten()[:15])
    elif isinstance(value, np.ndarray):
        print("Shape:", value.shape)
        print("First values:", value.flatten()[:15])

print("\nINSPECTION COMPLETED")

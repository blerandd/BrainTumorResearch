# Brain Tumor MRI Research

Research project for patient-independent brain tumor classification, tumor segmentation, and explainability using the Figshare brain tumor MRI dataset.

## Research tasks

- Classification: glioma, meningioma, and pituitary tumor.
- Classification comparison: Custom CNN, VGG16, and ResNet50.
- Segmentation: U-Net using the provided binary tumor masks.
- Explainability: Grad-CAM, compared descriptively and quantitatively with manual tumor masks.

## Dataset

Original Figshare record: https://doi.org/10.6084/m9.figshare.1512427

The dataset contains 3,064 T1-weighted contrast-enhanced MRI slices from 233 patients. It includes image-level labels, patient identifiers, and tumor masks. The dataset is distributed under CC BY 4.0. Cite the original dataset and follow its license terms in any report or publication.

## Environment

Use Python 3.10 or 3.11 in a virtual environment. TensorFlow installation can differ by operating system and processor, so verify the TensorFlow installation before training.

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Windows PowerShell

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Initial setup

1. Install dependencies.
2. Download and extract the official Figshare files:

```bash
python -m src.data.download_figshare
```

3. Convert MATLAB files into image, mask, and metadata files:

```bash
python -m src.data.prepare_dataset
```

4. Create patient-independent train/validation/test partitions:

```bash
python -m src.data.make_splits
```

The split script stratifies at patient level. It never intentionally separates slices from the same patient. Because patients contribute different numbers of slices, the final image proportions may differ from exactly 70/15/15. Always inspect the generated split report.

## Project structure

```text
BrainTumorResearch/
├── data/
│   ├── raw/
│   └── processed/
├── results/
│   ├── figures/
│   ├── tables/
│   └── checkpoints/
├── src/
│   ├── data/
│   └── models/
├── notebooks/
├── requirements.txt
└── README.md
```

## Reproducibility and evaluation rules

- Keep all slices from one patient in one partition.
- Apply augmentation to training data only.
- Use validation data for model selection and early stopping.
- Do not tune models using the test set.
- Report patient counts and image counts for every partition.
- Report per-class precision, recall, F1 and specificity, plus macro and weighted averages.
- Report Dice and IoU for segmentation.
- Record seeds, package versions, preprocessing settings, model checkpoints and training times.

## Important limitation

This is a research and educational project, not a clinical diagnostic system. Performance on this dataset does not establish clinical safety or generalization to other hospitals, scanners, populations, or MRI protocols.

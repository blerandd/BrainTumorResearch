# Brain Tumor MRI Research

A patient-independent deep learning research project for brain tumor MRI classification, tumor segmentation, and model explainability using the Figshare brain tumor dataset.

## Research objectives

This project investigates three related tasks:

- **Tumor classification:** classify MRI slices into glioma, meningioma, and pituitary tumor categories.
- **Tumor segmentation:** identify tumor regions using U-Net models and the dataset's manually annotated binary masks.
- **Explainability:** use Grad-CAM to visualize image regions that influence the classification model's predictions.

The classification experiments include a custom CNN, VGG16, and ResNet50. Segmentation experiments compare baseline U-Net and augmented U-Net variants, including a flip-only experiment.

## Dataset

**Source:** [Figshare – Brain Tumor Dataset](https://doi.org/10.6084/m9.figshare.1512427)

The dataset contains 3,064 contrast-enhanced T1-weighted MRI slices from 233 patients, with three tumor classes: glioma, meningioma, and pituitary. It provides patient identifiers, image-level class labels, and binary tumor masks. The original dataset is distributed under the **CC BY 4.0** license. Cite the original dataset and follow its license terms when using or publishing results.

### Data preparation and expected local files

The raw dataset and processed images are not stored in this repository. Download the original dataset from Figshare and prepare the local data before running the training or evaluation scripts.

The data loader expects this structure:

```text
data/
├── raw/
└── processed/
    ├── images/
    ├── masks/
    └── splits.csv
```

The `splits.csv` file must contain `sample_id`, `patient_id`, `label`, `image_path`, `mask_path`, and `split` columns. The `split` column must contain `train`, `validation`, and `test`. Image and mask paths are relative to `data/processed/`. Each patient must belong to exactly one partition; the dataset loader checks for patient leakage.

The repository includes `src/data/prepare_dataset.py` for converting original MATLAB files into PNG images, binary masks, and metadata. It expects the original `.mat` files under `data/raw/`. The resulting metadata must be organized into the `splits.csv` format required by the loader before training.

## Environment and installation

The project uses Python and TensorFlow. Dependencies are listed in `requirements.txt`. TensorFlow installation may vary by operating system and hardware.

### macOS / Linux

```bash
git clone https://github.com/blerandd/BrainTumorResearch.git
cd BrainTumorResearch

python3 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Windows PowerShell

```powershell
git clone https://github.com/blerandd/BrainTumorResearch.git
cd BrainTumorResearch

py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
pip install -r requirements.txt
```

Verify TensorFlow installation:

```bash
python -c "import tensorflow as tf; print(tf.__version__)"
```

## Running the project

Run commands from the repository root with the virtual environment activated and the local dataset prepared.

### Dataset inspection

```bash
python -m src.data.dataset_loader
python -m src.data.analyze_dataset
```

### Model training

```bash
python -m src.training.train_classification
python -m src.training.train_unet
python -m src.training.train_unet_augmented
python -m src.training.train_unet_flip_only
```

Training scripts save model checkpoints under `models/` and training histories, logs, and plots under `results/`.

### Evaluation

```bash
python -m src.evaluation.evaluate_classification
python -m src.evaluation.evaluate_unet
python -m src.evaluation.evaluate_patient_level
python -m src.evaluation.evaluate_boundary_metrics
```

Additional analysis scripts in `src/evaluation/` compare models, examine class-wise and patient-level performance, analyze segmentation errors, and calculate statistical and boundary-based metrics. Run an analysis module with `python -m src.evaluation.<module_name>` from the repository root.

### Grad-CAM explainability

```bash
python -m src.explainability.gradcam
```

Grad-CAM outputs and analysis files are saved under `results/explainability/`.

## Project structure

```text
BrainTumorResearch/
├── data/
│   ├── raw/                  # Original dataset files (not tracked)
│   └── processed/            # Images, masks, and split metadata (not tracked)
├── models/
│   ├── classification/       # Trained classification models (Git LFS)
│   └── segmentation/         # Trained U-Net models (Git LFS)
├── results/
│   ├── classification/
│   ├── explainability/
│   ├── figures/
│   └── segmentation/
├── src/
│   ├── data/                 # Dataset preparation, loading, and inspection
│   ├── evaluation/           # Metrics, comparisons, and statistical analyses
│   ├── explainability/       # Grad-CAM
│   ├── models/               # CNN and U-Net architectures
│   └── training/             # Model training pipelines
├── .gitattributes
├── .gitignore
├── requirements.txt
└── README.md
```

## Evaluation methodology

The project emphasizes patient-independent evaluation:

- Keep all MRI slices from a patient in the same data partition.
- Apply data augmentation to training data only.
- Use validation data for model selection and early stopping.
- Keep the test set separate from model selection and threshold tuning.
- Report classification performance per class and with macro/weighted averages.
- Evaluate segmentation using Dice, Intersection over Union (IoU), precision, and recall.
- Include patient-level summaries so results are not interpreted only from individual MRI slices.
- Use boundary metrics and qualitative visualizations to examine segmentation quality and failure cases.
- Record random seeds, preprocessing settings, package versions, and model configurations for reproducibility.

## Models and outputs

Trained model checkpoints are stored in Git LFS:

- Custom CNN: `models/classification/custom_cnn_best.keras`
- Baseline U-Net: `models/segmentation/unet_best.keras`
- Augmented U-Net: `models/segmentation/unet_augmented_best.keras`
- Flip-only U-Net: `models/segmentation/unet_flip_only_best.keras`

Evaluation tables, plots, Grad-CAM visualizations, and experiment comparisons are organized under `results/`.

## Limitations and responsible use

This is an academic research project, not a medical device or clinical diagnostic system. Results on this dataset do not establish clinical safety or generalization to other hospitals, scanners, patient populations, or MRI protocols. Model predictions must not be used as a substitute for assessment by qualified medical professionals.

## Citation

If you use the dataset, cite the original Figshare record:

[Brain Tumor Dataset – Figshare](https://doi.org/10.6084/m9.figshare.1512427)

Consult the dataset record for its authorship, recommended citation, and license information.

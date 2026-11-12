# Brain Tumor Research — Documentation

## 1. About This Documentation

This directory contains the technical and research documentation for the **BrainTumorResearch** project.

The purpose of this documentation is to provide a comprehensive explanation of the research methodology, dataset, preprocessing procedures, model architectures, training experiments, evaluation methods, statistical analysis, explainability techniques, and experimental findings.

The documentation is divided into individual chapters, with each document focusing on a specific aspect of the research.

It is intended to make the project easier to understand, evaluate, maintain, and reproduce.

## 2. Project Overview

BrainTumorResearch is a deep learning research project focused on brain tumor MRI image classification and tumor segmentation.

The project investigates the application of convolutional neural networks and deep learning architectures to medical image analysis.

The research focuses on three main areas:

- **Classification:** Identifying brain tumor types from MRI images.
- **Segmentation:** Identifying and outlining tumor regions in MRI images.
- **Explainability:** Visualizing which image regions contribute to classification model predictions.

The classification task considers three tumor categories:

- Glioma
- Meningioma
- Pituitary tumor

In addition, the research investigates data augmentation, patient-independent dataset splitting, model performance, patient-level evaluation, statistical comparisons, and failure case analysis.

## 3. Documentation Structure

The documentation is organized into 14 chapters.

| File | Description |
|---|---|
| Chapter | Description |
|---|---|
| [01 — Project Overview](01_project_overview.md) | Research objectives, problem definition, scope, and methodology. |
| [02 — Dataset and Preprocessing](02_dataset_and_preprocessing.md) | Dataset source, organization, preprocessing, and patient-level data splitting. |
| [03 — Exploratory Data Analysis](03_exploratory_data_analysis.md) | Dataset exploration, class distribution, patient distribution, and visual analysis. |
| [04 — Classification Models](04_classification_models.md) | Classification architectures, model configurations, and implementation. |
| [05 — Segmentation Models](05_segmentation_models.md) | U-Net architecture and segmentation model variants. |
| [06 — Training and Experiments](06_training_and_experiments.md) | Training procedures, experimental configurations, and augmentation strategies. |
| [07 — Evaluation Metrics](07_evaluation_metrics.md) | Classification and segmentation evaluation metrics. |
| [08 — Patient-Level Analysis](08_patient_level_analysis.md) | Patient-level evaluation and performance analysis across individual patients. |
| [09 — Statistical Analysis](09_statistical_analysis.md) | Paired comparisons, confidence intervals, p-values, and Holm correction. |
| [10 — Explainability with Grad-CAM](10_explainability_gradcam.md) | Grad-CAM methodology and classification model explainability. |
| [11 — Failure-Case Analysis](11_failure_case_analysis.md) | Incorrect classification predictions and segmentation failure cases. |
| [12 — Results and Discussion](12_results_and_discussion.md) | Main experimental results, interpretation, and research findings. |
| [13 — Reproducibility and Environment](13_reproducibility_and_environment.md) | Software requirements, environment configuration, and experimental reproducibility. |
| [14 — Limitations and Future Work](14_limitations_and_future_work.md) | Research limitations and possible future improvements. |

## 4. Photos and Visual Evidence

The documentation image library is organized under `Docs/photos/` (relative to the repository root) and contains selected visual materials supporting the written chapters. This library is distinct from `results/`, which stores generated experimental outputs such as evaluation plots, CSV files, and Grad-CAM results.

These images provide visual evidence of the dataset, preprocessing pipeline, model training, experimental results, and evaluation procedures.

The images are organized into the following categories:

| Directory | Contents |
|---|---|
| `photos/dataset/` | Original MRI dataset examples and sample images. |
| `photos/preprocessing/` | Image preprocessing, resizing, normalization, and mask preparation examples. |
| `photos/classification/` | Classification predictions and classification-related visualizations. |
| `photos/segmentation/` | MRI images, ground-truth masks, and predicted segmentation masks. |
| `photos/training/` | Training and validation loss and accuracy graphs. |
| `photos/model_comparison/` | Comparisons between different classification and segmentation models. |
| `photos/gradcam/` | Grad-CAM heatmaps and explainability examples. |
| `photos/failure_analysis/` | Incorrect predictions and segmentation failure examples. |
| `photos/statistical_analysis/` | Statistical comparison plots, confidence intervals, and patient-level analysis figures. |

Only selected images relevant to the research should be included.

Where possible, visualizations should be taken directly from the project's saved experimental outputs.

### 4.1 Referencing Images in Markdown

Images can be displayed directly inside chapter Markdown files using paths relative to the `Docs/` directory. For example, a file stored at `Docs/photos/dataset/example.png` is referenced from a chapter as:

For example:

```markdown
![Brain MRI example](photos/dataset/example.png)
```

For a segmentation comparison:

```markdown
![U-Net segmentation results](photos/segmentation/example.png)
```

Using relative paths allows images to display correctly when the repository is viewed on GitHub.

## 5. Dataset Information

The research uses the publicly available brain tumor MRI dataset hosted on Figshare.

**Dataset:** Brain Tumor MRI Dataset

**Source:** https://doi.org/10.6084/m9.figshare.1512427

The dataset contains 3,064 MRI image slices from 233 patients.

It includes three tumor categories:

- Glioma: 1,426 images.
- Meningioma: 708 images.
- Pituitary tumor: 930 images.

The project uses patient-independent data splitting to reduce the risk of data leakage between training, validation, and testing.

The recorded dataset split proportions are:

| Partition | Proportion |
|---|---:|
| Training | 70% |
| Validation | 15% |
| Testing | 15% |

The dataset is not included directly in this documentation directory.

Refer to `02_dataset_and_preprocessing.md` for further information about dataset acquisition, organization, preprocessing, and splitting.

## 6. Research Methodology

The research follows a structured experimental workflow:

1. Dataset acquisition and exploration.
2. Data preprocessing and preparation.
3. Patient-independent dataset splitting.
4. Development and training of classification models.
5. Development and training of U-Net segmentation models.
6. Evaluation of classification and segmentation performance.
7. Patient-level performance analysis.
8. Statistical comparison of segmentation experiments.
9. Grad-CAM explainability analysis.
10. Failure case analysis.
11. Interpretation and discussion of experimental results.

This methodology combines quantitative evaluation with visual and statistical analysis.

## 7. Technologies and Tools

The project uses Python and scientific computing and deep learning libraries for model development, training, and evaluation.

The main technologies and libraries include:

- **Python** — Main programming language.
- **TensorFlow** — Deep learning framework.
- **Keras** — Neural network development and training API.
- **NumPy** — Numerical computing and array manipulation.
- **Pandas** — Data organization and analysis.
- **Scikit-learn** — Machine learning evaluation and statistical utilities.
- **Matplotlib** — Data visualization.
- **Seaborn** — Statistical visualization.
- **OpenCV** — Image processing.
- **Pillow** — Image loading and manipulation.

The complete software requirements and version constraints are maintained in the repository-root [`requirements.txt`](../requirements.txt). Use that file as the authoritative dependency list rather than copying package versions into this documentation index.

## 8. Main Experimental Findings

The project investigates classification performance and three U-Net segmentation configurations.

The recorded classification evaluation includes:

| Metric | Result |
|---|---:|
| Test images | 489 |
| Accuracy | 0.8466 |
| Macro ROC-AUC | 0.9598 |

The recorded image-level mean Dice scores for the segmentation experiments are:

| Segmentation Configuration | Mean Dice |
|---|---:|
| Baseline U-Net | 0.7016 |
| Flip-only U-Net | 0.7259 |
| Flip and rotation U-Net | 0.7281 |

The patient-level statistical comparison includes 35 patients.

The recorded comparison between flip-only U-Net and baseline U-Net produced:

| Measurement | Result |
|---|---:|
| Mean paired Dice difference | +0.0403 |
| Confidence interval | [0.0103, 0.0761] |
| Holm-adjusted p-value | 0.0019 |

The flip-only configuration showed a statistically significant improvement over the baseline in the reported paired patient-level comparison.

The comparisons involving flip and rotation did not establish statistically significant differences.

These values are documented in greater detail in the corresponding evaluation, statistical analysis, and results chapters.

## 9. How to Navigate This Documentation

For readers who want to understand the research from beginning to end, the recommended reading order is:

1. **Project foundations:** Chapters 01–03.
2. **Model architectures:** Chapters 04–05.
3. **Training and experiments:** Chapter 06.
4. **Quantitative evaluation:** Chapter 07.
5. **Patient-level and statistical analysis:** Chapters 08–09.
6. **Explainability and failure analysis:** Chapters 10–11.
7. **Experimental results and discussion:** Chapter 12.
8. **Reproducibility and research limitations:** Chapters 13–14.

Readers interested in a particular research component can also navigate directly to its corresponding chapter.

## 10. Relationship with the Main Repository

This documentation is part of the main BrainTumorResearch repository.

The main repository contains the implementation and experimental resources, including:

- Dataset preparation scripts.
- Image preprocessing procedures.
- Classification model implementations.
- Segmentation model implementations.
- Training scripts.
- Evaluation scripts.
- Statistical analysis scripts.
- Saved model checkpoints.
- Experimental results and visualizations.

The `Docs/` directory (capital D) provides the detailed written explanation of the research methodology, implementation, and findings. Preserve this capitalization in links and paths, especially on case-sensitive file systems.

For the general project introduction, installation information, and main repository overview, refer to the [root README](../README.md). For the complete source code, environment dependencies, checkpoints, and generated experiment outputs, navigate to the corresponding directories in the repository root.

## 11. Documentation Maintenance

The documentation should be updated whenever important changes are made to the research implementation.

Examples include:

- Adding a new model architecture.
- Changing preprocessing configurations.
- Updating dataset partitions.
- Performing additional experiments.
- Generating new evaluation results.
- Adding statistical comparisons.
- Introducing new explainability methods.

Experimental results should always be checked against the corresponding saved outputs and implementation before being added to the documentation.

This helps maintain consistency between the written research and the actual repository.

---

**Project:** BrainTumorResearch  
**Documentation:** Technical and Research Documentation  
**Number of Chapters:** 14  
**Dataset:** Figshare Brain Tumor MRI Dataset  
**Main Programming Language:** Python  
**Research Areas:** Deep Learning, Medical Image Classification, Image Segmentation, Statistical Analysis, and Explainable AI

## 12. Implementation Details Verified Against the Source Code

The following details were checked against the current Python implementation:

- The dataset loader validates required metadata columns, missing values, class labels, split names, image and mask file existence, and patient overlap across partitions.
- MRI images are decoded as single-channel unsigned 16-bit PNG data, converted to float32, divided by 65535.0, and resized to 256 × 256 using bilinear interpolation.
- Segmentation masks are decoded as single-channel unsigned 8-bit PNG data, converted to binary values, resized using nearest-neighbor interpolation, and thresholded again.
- The U-Net has encoder stages with 32, 64, 128, and 256 filters, a 512-filter bottleneck, transposed-convolution upsampling, skip connections, and a one-channel sigmoid output.
- The U-Net uses Adam with learning rate 0.0001 and a combined binary-cross-entropy plus Dice loss.
- The baseline U-Net training script specifies 50 maximum epochs, batch size 4, and seed 42. It uses best-checkpoint saving on validation loss, early stopping, and learning-rate reduction on plateau.
- The standard U-Net evaluation thresholds predicted probabilities at 0.5 and calculates per-image Dice, IoU, pixel precision, and pixel recall. Patient summaries average image metrics within each patient.

These settings describe the baseline implementation and should not automatically be attributed to every experiment. The flip-only and flip-plus-rotation training scripts must be checked for their own configurations.

### Documentation path convention

The directory name in this repository is Docs/ (capital D). Relative links and file paths should preserve that capitalization on case-sensitive systems.


## 13. Documentation and Results Path Convention

Use these repository-relative locations consistently:

- `Docs/` — written technical and research chapters.
- `Docs/photos/` — selected supporting images embedded in documentation chapters.
- `results/` — generated experimental outputs, including metrics, CSV files, plots, and explainability artifacts.
- `models/` — saved model checkpoints.
- `requirements.txt` — Python dependency constraints for the project environment.

The root [README](../README.md) is the short project entry point. This file is the index for the detailed research documentation. Keep chapter links relative to this directory and links to repository-root resources prefixed with `../`.

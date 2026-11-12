# 13 — Reproducibility and Environment

## 1. Introduction

Reproducibility is an important requirement in scientific research and machine learning.

A research experiment should be documented in enough detail that another researcher can understand the environment, dataset preparation, model configurations, training procedures, and evaluation methods.

BrainTumorResearch uses Python-based deep learning tools to investigate brain tumor classification and segmentation using MRI images.

This chapter documents the development environment, software dependencies, repository organization, and experimental workflow.

It also describes important considerations for reproducing the classification and segmentation experiments.

## 2. Development Environment

The project was developed and tested using a local development environment on an Apple Silicon Mac.

The recorded environment includes:

| Component | Recorded Configuration |
|---|---|
| Computer | Apple Mac with M2 chip |
| Processor architecture | ARM64 |
| Development environment | PyCharm |
| Programming language | Python |
| Python version previously used | 3.9.6 |
| Deep learning framework | TensorFlow 2.21.0 |
| Numerical computing library | NumPy 2.4.6 |
| Virtual environment | `.venv` |
| GPU availability | No GPU detected by TensorFlow |

These values describe the previously recorded development environment.

Software versions can change over time, so the current environment should be checked before reproducing the experiments.

## 3. Hardware Environment

The project was developed on an Apple M2 processor.

Apple Silicon uses the ARM64 architecture.

This is relevant because some Python packages and machine learning libraries provide architecture-specific installations.

During the recorded environment checks, TensorFlow reported CPU availability but did not detect a GPU.

Therefore, the experiments were run without a TensorFlow GPU device being detected.

### 3.1 CPU Execution

When TensorFlow does not detect a supported GPU, model operations can execute on the CPU.

This allows deep learning experiments to run without a dedicated NVIDIA GPU.

However, training performance may differ depending on:

- Processor architecture.
- Available memory.
- TensorFlow version.
- Model complexity.
- Batch size.
- Image resolution.

### 3.2 Apple Silicon Considerations

Apple Silicon environments may use different acceleration configurations from NVIDIA CUDA environments.

The absence of an NVIDIA GPU does not automatically mean that Apple hardware acceleration is unavailable.

The actual available TensorFlow devices should be checked in the active Python environment.

For this project, the previously recorded device check returned an empty GPU list.

## 4. Python Virtual Environment

The project uses a Python virtual environment named:

`.venv`

A virtual environment creates an isolated Python environment for a project.

It helps prevent package conflicts between different projects.

It also allows the project's dependencies to be installed separately from the system-wide Python installation.

### 4.1 Creating a Virtual Environment

From the project root directory, a virtual environment can be created using:

```bash
python3 -m venv .venv
```

### 4.2 Activating the Environment on macOS

Use:

```bash
source .venv/bin/activate
```

After activation, the terminal should indicate that the virtual environment is active.

### 4.3 Checking the Python Version

Use:

```bash
python --version
```

### 4.4 Checking the Python Executable

Use:

```bash
which python
```

The returned path should point to the project's `.venv` directory when the virtual environment is active.

### 4.5 Installing Dependencies

If the repository contains a `requirements.txt` file, dependencies can be installed using:

```bash
pip install -r requirements.txt
```

The dependency file should represent the packages and versions required by the actual project.

Do not assume that every package installed in the development environment is necessary for reproducing the experiments.

## 5. Main Software Technologies

The project uses Python and several scientific computing and deep learning libraries.

### 5.1 Python

Python is the main programming language used for:

- Dataset preparation.
- Image preprocessing.
- Exploratory data analysis.
- Model development.
- Model training.
- Model evaluation.
- Statistical analysis.
- Visualization.

### 5.2 TensorFlow

TensorFlow is the main deep learning framework used in the project.

It supports:

- Neural network construction.
- Model training.
- Automatic differentiation.
- Tensor operations.
- Model saving and loading.
- Evaluation workflows.

The recorded environment used TensorFlow 2.21.0.

### 5.3 NumPy

NumPy is used for numerical operations and array manipulation.

It supports image processing, mathematical calculations, and the handling of numerical data.

The recorded environment used NumPy 2.4.6.

### 5.4 Other Libraries

The project also uses Python libraries for supporting tasks such as:

- Image loading and processing.
- Data organization.
- Statistical analysis.
- Plotting and visualization.
- Evaluation metrics.

The exact dependency versions should be taken from the repository's dependency files or the saved environment information.

## 6. Checking the Environment

The following commands can be used to inspect the active environment.

### Python Version

```bash
python --version
```

### TensorFlow Version

```bash
python -c "import tensorflow as tf; print(tf.__version__)"
```

### NumPy Version

```bash
python -c "import numpy as np; print(np.__version__)"
```

### Processor Architecture

```bash
python -c "import platform; print(platform.machine())"
```

### TensorFlow Available Devices

```bash
python -c "import tensorflow as tf; print(tf.config.list_physical_devices())"
```

### Available GPU Devices

```bash
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

These commands help verify whether the environment matches the recorded development configuration.

## 7. Dataset Reproducibility

The project uses the brain tumor MRI dataset available through Figshare.

Dataset reference:

https://doi.org/10.6084/m9.figshare.1512427

The dataset contains:

- 3,064 MRI images.
- 233 patients.
- Three tumor categories.

The categories are:

1. Glioma.
2. Meningioma.
3. Pituitary tumor.

Dataset preparation is an important part of reproducibility because the same model can produce different results when trained on different data partitions or preprocessing configurations.

The dataset preparation process is documented in:

`02_dataset_and_preprocessing.md`

## 8. Dataset Splitting and Random Seed

The project uses a patient-independent dataset splitting strategy.

The recorded proportions are:

| Partition | Proportion |
|---|---:|
| Training | 70% |
| Validation | 15% |
| Testing | 15% |

The recorded random seed is:

```python
seed = 42
```

Using a fixed seed helps make randomized operations more consistent between runs.

Patient-independent splitting is particularly important because multiple MRI images may belong to the same individual.

Images from one patient should not be distributed across training and testing partitions.

Otherwise, the model could be evaluated on images from a patient whose other images were already used during training.

The actual split assignments should be preserved when comparing experimental configurations.

## 9. Image Preprocessing Configuration

The recorded dataset loader uses the following configuration:

| Parameter | Value |
|---|---|
| Image height | 256 pixels |
| Image width | 256 pixels |
| Number of channels | 1 |
| Batch size | 16 |
| Image data type before normalization | `uint16` |
| Image data type after normalization | `float32` |
| Random seed | 42 |

The dataset loader processes PNG images.

The image values are converted from unsigned 16-bit integers into floating-point values.

Normalization is performed using:

\[
I_{normalized} = \frac{I}{65535.0}
\]

The resulting values are scaled to the interval:

\[
[0,1]
\]

Images are resized to:

\[
256 \times 256
\]

The dataset loader is located at:

`src/data/dataset_loader.py`

The preprocessing pipeline is described in detail in:

`02_dataset_and_preprocessing.md`

### 9.1 Segmentation Masks

Segmentation masks require special handling during resizing.

Unlike ordinary MRI images, masks contain discrete class or region labels.

Nearest-neighbor interpolation is used for mask resizing to avoid introducing artificial intermediate label values.

The preprocessing procedure should preserve the correspondence between each MRI image and its ground-truth segmentation mask.

## 10. Repository Organization

The repository separates source code, datasets, trained models, results, and documentation.

The main directories include:

| Directory | Purpose |
|---|---|
| `data/` | Dataset and processed data |
| `models/` | Saved trained models |
| `src/` | Source code |
| `results/` | Evaluation results and visualizations |
| `docs/` | Research documentation |

The source code is organized into functional areas, including:

- Data processing.
- Model development.
- Model training.
- Evaluation.
- Statistical analysis.

This organization helps separate experimental implementation from generated outputs.

The complete repository structure should be consulted when locating individual scripts or model checkpoints.

## 11. Model Checkpoints and Reproducibility

Trained model checkpoints allow previously trained models to be loaded without repeating the complete training process.

The project contains classification and segmentation checkpoints.

A recorded classification checkpoint is:

`models/classification/custom_cnn_best.keras`

The segmentation checkpoints include:

- `models/segmentation/unet_best.keras`
- `models/segmentation/unet_flip_only_best.keras`
- `models/segmentation/unet_augmented_best.keras`

The checkpoints represent saved model states from the corresponding experiments.

When evaluating a saved checkpoint, it is important to use compatible preprocessing and evaluation configurations.

A checkpoint should not be assumed to reproduce an experiment if its architecture, class mapping, preprocessing, or model configuration is different.

## 12. Reproducing the Classification Experiments

The classification workflow can be understood through the following stages:

1. Prepare the MRI dataset.
2. Organize the training, validation, and testing partitions.
3. Apply the required preprocessing.
4. Load the classification dataset.
5. Construct the selected CNN architecture.
6. Train the model using the training partition.
7. Monitor performance using the validation partition.
8. Save the best model checkpoint.
9. Evaluate the selected model using the test partition.
10. Generate classification metrics and visualizations.

The classification experiments investigate Custom CNN, VGG16, and ResNet50 architectures.

The exact training configuration for each architecture should be taken from its corresponding training script.

The classification methodology is documented in:

`04_classification_models.md`

## 13. Reproducing the Segmentation Experiments

The segmentation workflow follows a similar structure.

The main stages are:

1. Prepare MRI images and corresponding masks.
2. Apply image preprocessing.
3. Resize MRI images and segmentation masks.
4. Load the training, validation, and testing partitions.
5. Construct the U-Net architecture.
6. Train the selected configuration.
7. Save the best model checkpoint.
8. Generate segmentation predictions.
9. Compare predicted masks with ground-truth masks.
10. Calculate segmentation metrics.
11. Generate evaluation visualizations.

The project investigates three configurations:

- Baseline U-Net.
- Flip-only U-Net.
- Flip and rotation U-Net.

The training and augmentation procedures are documented in:

`05_segmentation_models.md`

`06_training_and_experiments.md`

## 14. Reproducing Evaluation Results

The evaluation stage uses saved checkpoints and the appropriate test data.

Classification evaluation includes metrics such as:

- Accuracy.
- Precision.
- Recall.
- F1-score.
- Confusion matrix.
- ROC-AUC.

Segmentation evaluation includes:

- Dice coefficient.
- Intersection over Union.
- Boundary-related metrics.
- Patient-level performance summaries.

The evaluation methodology is documented in:

`07_evaluation_metrics.md`

Patient-level analysis is documented in:

`08_patient_level_analysis.md`

Statistical comparisons are documented in:

`09_statistical_analysis.md`

The preprocessing and evaluation configurations must remain consistent when comparing different model checkpoints.

## 15. Reproducibility Considerations

Even when the same source code and random seed are used, machine learning experiments may not produce perfectly identical results.

Differences can occur because of:

- Hardware architecture.
- Software versions.
- Numerical precision.
- TensorFlow implementation details.
- Randomized operations.
- Parallel execution.
- Different package configurations.

Therefore, reproducibility should be understood as the ability to reconstruct the experimental setup and obtain comparable results, rather than a guarantee of identical numerical output in every environment.

For stronger reproducibility, the project should preserve:

- Dataset version and source.
- Dataset split assignments.
- Random seed.
- Python version.
- Dependency versions.
- Model architecture.
- Training configuration.
- Model checkpoints.
- Evaluation scripts.
- Generated results.

## 16. Environment Limitations

The recorded development environment has several characteristics that should be considered.

### 16.1 Hardware Differences

The project was developed on Apple M2 hardware.

A researcher using a different processor or GPU may observe different training speeds and numerical results.

### 16.2 GPU Availability

The recorded TensorFlow environment did not detect a GPU.

Consequently, the recorded experiments should not be described as NVIDIA CUDA-based GPU experiments.

### 16.3 Software Compatibility

Python and deep learning libraries evolve over time.

Installing different versions of TensorFlow, NumPy, or other dependencies may affect compatibility.

The actual working dependency versions should therefore be documented.

### 16.4 Computational Requirements

Training CNN and U-Net architectures requires computational resources.

The required time and memory depend on:

- Model architecture.
- Image resolution.
- Batch size.
- Number of epochs.
- Available hardware.

## 17. Recommendations for Reproducible Research

To make the project easier to reproduce, the following practices are recommended:

1. Preserve the original dataset source and metadata.
2. Keep patient-level dataset assignments unchanged during model comparisons.
3. Record the random seed.
4. Document the exact Python and package versions.
5. Preserve the model training scripts.
6. Save the best model checkpoints.
7. Keep evaluation scripts and generated metrics.
8. Record the configuration used for each experiment.
9. Document hardware and available acceleration devices.
10. Avoid changing preprocessing between model comparisons without recording the change.

These practices make it easier to understand how experimental results were produced.

## 18. Chapter Summary

This chapter documented the software and hardware environment used in BrainTumorResearch and described the procedures needed to understand and reproduce the experiments.

The project uses Python, TensorFlow, NumPy, and a Python virtual environment.

The recorded development environment includes an Apple M2 processor, ARM64 architecture, TensorFlow 2.21.0, NumPy 2.4.6, and a `.venv` environment.

The dataset contains 3,064 MRI images from 233 patients and uses a patient-independent 70/15/15 split with random seed 42.

The project includes separate workflows for classification and segmentation, together with saved model checkpoints, evaluation scripts, statistical outputs, and visualizations.

Reproducibility depends on preserving the dataset, preprocessing configuration, software dependencies, training procedures, and evaluation methods.

The following chapter focuses on the limitations of the research and possible directions for future work.

---

**Document:** 13 — Reproducibility and Environment  
**Project:** BrainTumorResearch  
**Development Hardware:** Apple M2  
**Architecture:** ARM64  
**Deep Learning Framework:** TensorFlow 2.21.0 (recorded environment)  
**Numerical Library:** NumPy 2.4.6 (recorded environment)  
**Virtual Environment:** `.venv`  
**Dataset:** Figshare Brain Tumor MRI Dataset  
**Dataset Split:** 70% / 15% / 15%  
**Random Seed:** 42

## Source-Code Configuration Checks

The following settings were verified against the current scripts:

| Component | Verified configuration |
|---|---|
| Dataset loader defaults | Image size 256 × 256; batch size 16; seed 42 |
| Baseline U-Net training | Maximum 50 epochs; batch size 4; seed 42 |
| Flip-only U-Net training | Maximum 50 epochs; batch size 4; seed 42; horizontal flip with probability 0.5 |
| Flip-plus-rotation U-Net training | Maximum 50 epochs; batch size 4; seed 42; horizontal flip probability 0.5; random rotation between -10 and +10 degrees |
| Classification evaluation | Custom CNN checkpoint; test split; batch size 16; seed 42 |
| Grad-CAM | Custom CNN checkpoint; target layer activation_3; heatmap resized to 256 × 256 |

For the rotation experiment, the MRI image is transformed using bilinear interpolation while its mask is transformed using nearest-neighbor interpolation. This preserves smoother image intensities while avoiding interpolated class labels in the binary mask.

A configured maximum epoch count is not necessarily the number of epochs actually completed. Early stopping can stop training sooner. Reproduction should therefore use the saved training log and history CSV to report the executed epoch count and best epoch, rather than reporting the configured maximum as the actual duration.

The local environment recorded for development is machine-specific. TensorFlow device availability can differ across computers, so GPU acceleration should be reported from the actual runtime rather than inferred from the presence of TensorFlow or an Apple Silicon processor.


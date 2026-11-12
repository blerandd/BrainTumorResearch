# 02 — Dataset and Preprocessing

## 1. Introduction

This chapter describes the dataset used in the BrainTumorResearch project and the preprocessing procedures applied before training and evaluating the deep learning models.

The quality and organization of medical imaging data are important factors in deep learning research. MRI images must be loaded, inspected, prepared, and transformed into a consistent format before they can be used by neural networks.

The project uses a publicly available brain tumor MRI dataset hosted on Figshare.

The dataset is used for two main tasks:

- Brain tumor classification.
- Brain tumor segmentation.

This chapter explains the dataset source, image categories, patient organization, preprocessing pipeline, segmentation mask preparation, and patient-independent dataset splitting.

## 2. Dataset Source

The research uses the publicly available Brain Tumor MRI dataset hosted on Figshare.

**Dataset:** Brain Tumor MRI Dataset

**Source:** https://doi.org/10.6084/m9.figshare.1512427

The dataset contains MRI image slices associated with three brain tumor categories.

The dataset provides MRI images and corresponding segmentation information that can be used for classification and segmentation experiments.

The project uses the dataset as the primary source of image data for training, validation, and testing.

## 3. Dataset Overview

The dataset contains a total of:

- **3,064 MRI image slices**
- **233 patients**
- **3 tumor categories**

The distribution of MRI images across the three categories is:

| Tumor Category | Number of Images |
|---|---:|
| Glioma | 1,426 |
| Meningioma | 708 |
| Pituitary tumor | 930 |
| **Total** | **3,064** |

### 3.1 Glioma

Glioma is a type of brain tumor that develops from glial cells.

The dataset contains 1,426 MRI images belonging to the glioma category.

### 3.2 Meningioma

Meningioma develops from the meninges, the membranes surrounding the brain and spinal cord.

The dataset contains 708 MRI images belonging to the meningioma category.

### 3.3 Pituitary Tumor

Pituitary tumors develop in or around the pituitary gland.

The dataset contains 930 MRI images belonging to the pituitary tumor category.

## 4. Patient-Level Organization

An important characteristic of the dataset is that MRI images are associated with individual patients.

A single patient may contribute multiple MRI slices to the dataset.

This means that the number of images is not equal to the number of independent patients.

The project therefore considers patient identity when dividing the dataset into training, validation, and testing partitions.

This is particularly important because images belonging to the same patient may share visual characteristics.

If images from the same patient appear in both training and testing partitions, the model may learn patient-specific patterns rather than general patterns that apply to previously unseen patients.

This can lead to overly optimistic evaluation results.

To reduce this risk, the project uses patient-independent data splitting.

## 5. Dataset Exploration

Before preprocessing and training, the dataset is explored to understand its structure and characteristics.

The exploration process includes:

- Counting the total number of MRI images.
- Examining the distribution of tumor categories.
- Investigating the number of images associated with each patient.
- Inspecting sample MRI images.
- Examining the availability of segmentation masks.
- Investigating tumor-region characteristics.

The project includes dataset analysis and visualization scripts in:

`src/data/`

Relevant scripts include:

- `analyze_dataset.py`
- `explore_dataset.py`
- `visualize_dataset.py`
- `analyze_tumor_intensity_patient.py`

These scripts support the inspection and analysis of the dataset before model training.

## 6. Image Loading

The project uses a TensorFlow-based data loading pipeline.

The dataset loader is implemented in:

`src/data/dataset_loader.py`

The loader is responsible for reading the prepared dataset, decoding images, processing image values, and preparing data for model input.

The model data pipeline uses the following configuration:

| Parameter | Value |
|---|---|
| Image height | 256 pixels |
| Image width | 256 pixels |
| Number of channels | 1 |
| Batch size | 16 |
| Random seed | 42 |
| Image data type during decoding | `uint16` |
| Model input data type | `float32` |

The three classification categories are:

- `glioma`
- `meningioma`
- `pituitary`

## 7. Image Resizing

MRI images may have different original dimensions.

Neural networks require a consistent input shape within a batch. Therefore, images are resized to a fixed spatial resolution.

The project uses:

**256 × 256 pixels**

This means that each image is represented using:

- 256 pixels in height.
- 256 pixels in width.
- 1 image channel.

The resulting input shape is:

`(256, 256, 1)`

Resizing provides a consistent input format for the classification and segmentation pipelines.

## 8. Image Normalization

The dataset loader decodes MRI images using the `uint16` data type.

These values are converted into floating-point values and normalized.

The normalization operation is:

\[
I_{normalized} = \frac{I_{original}}{65535.0}
\]

Where:

- \(I_{original}\) represents the original pixel intensity.
- \(65535\) is the maximum value representable by an unsigned 16-bit integer.
- \(I_{normalized}\) represents the normalized pixel intensity.

After normalization, the image values are represented as floating-point numbers in the interval:

\[
[0,1]
\]

The normalized image data is converted to `float32`.

This transformation provides a consistent numerical range for neural network processing.

## 9. Segmentation Mask Preparation

In addition to MRI images, the segmentation pipeline uses corresponding tumor masks.

A segmentation mask identifies the pixels belonging to the tumor region.

Unlike an MRI image, which contains intensity information, a segmentation mask represents the spatial region that the model is expected to identify.

The segmentation pipeline must preserve the correspondence between each MRI image and its associated mask.

### 9.1 Mask Resizing

Segmentation masks must be resized to the same spatial dimensions as their corresponding MRI images.

The project uses a target resolution of:

**256 × 256 pixels**

For segmentation masks, nearest-neighbor interpolation is used when resizing.

Nearest-neighbor interpolation is appropriate for categorical mask values because it avoids introducing new intermediate values between different mask labels.

This is important for preserving the distinction between background and tumor pixels.

### 9.2 Image and Mask Alignment

The MRI image and its corresponding segmentation mask must have matching spatial dimensions.

For each training example:

- The MRI image provides the input.
- The segmentation mask provides the target output.

The model learns to predict the tumor region based on the MRI image.

## 10. Dataset Splitting

The project uses a patient-independent dataset splitting strategy.

The dataset is divided into three partitions:

| Partition | Percentage | Purpose |
|---|---:|---|
| Training | 70% | Used to train the models. |
| Validation | 15% | Used to monitor performance and support model selection. |
| Testing | 15% | Used for final model evaluation. |

The splitting process uses a random seed of:

`42`

### 10.1 Training Set

The training set contains the images used to update model parameters during training.

The neural network learns patterns from these examples by minimizing its training loss.

### 10.2 Validation Set

The validation set is used to monitor model performance during training.

It helps evaluate whether the model generalizes to data that was not used for parameter updates.

Validation results can also support model selection and training decisions.

### 10.3 Testing Set

The testing set is reserved for final evaluation.

It is not used to update model parameters during training.

Its purpose is to provide an independent estimate of model performance on previously unseen patients.

### 10.4 Prevention of Patient Leakage

The most important requirement of the splitting procedure is that images belonging to the same patient must remain in the same partition.

For example, if a patient's MRI images are assigned to the training set, other images from that patient must not appear in the validation or testing sets.

This prevents information from the same patient from being shared across different partitions.

The project also includes checks for patient overlap between partitions.

The dataset splitting and preparation scripts are located in:

- `src/data/make_splits.py`
- `src/data/prepare_dataset.py`

## 11. Data Augmentation

Data augmentation is investigated in the segmentation experiments to examine whether transformations can improve model generalization.

The project includes different segmentation training configurations:

- Baseline U-Net.
- U-Net with horizontal and vertical flip augmentation.
- U-Net with flip and rotation augmentation.

Augmentation is applied as part of the training process.

For segmentation, transformations must be applied consistently to both the MRI image and its corresponding mask.

For example, if an image is flipped, its segmentation mask must be flipped in the same way.

Otherwise, the relationship between the input image and target mask would be incorrect.

The detailed augmentation experiments are described in:

`06_training_and_experiments.md`

## 12. Processed Dataset Organization

The data preparation pipeline uses the following processed dataset location:

`data/processed/`

The dataset loader also references a metadata file:

`splits.csv`

This file contains information used to identify the dataset partitions.

The processed dataset is used by the model training and evaluation pipelines.

The original and processed datasets are not stored inside the documentation directory.

The project repository excludes the raw and processed dataset directories from Git tracking.

## 13. Data Leakage Prevention

Data leakage occurs when information from the validation or testing data influences the model during training.

In medical imaging, patient-level leakage is particularly important because multiple images may originate from the same individual.

The project addresses this issue through patient-independent dataset partitioning.

The implementation includes checks that examine whether patient identifiers overlap between the training, validation, and testing partitions.

This helps ensure that final testing evaluates the model using patients that were not included in training.

## 14. Preprocessing Pipeline Summary

The preprocessing workflow can be summarized as follows:

1. Obtain the MRI dataset from Figshare.
2. Inspect the dataset structure and image categories.
3. Identify patient-level information.
4. Prepare the dataset metadata.
5. Create patient-independent training, validation, and testing partitions.
6. Load MRI images using the dataset loader.
7. Decode image data as `uint16`.
8. Resize images to 256 × 256 pixels.
9. Convert image values to `float32`.
10. Normalize pixel intensities by dividing by 65535.
11. Prepare segmentation masks using nearest-neighbor resizing.
12. Load the prepared data for classification and segmentation experiments.

## 15. Importance of the Preprocessing Stage

Preprocessing is an essential part of the research because it ensures that the input data is consistent and suitable for deep learning models.

The main benefits include:

- Consistent image dimensions.
- Consistent numerical input representation.
- Correct alignment between MRI images and segmentation masks.
- Reduced risk of patient-level data leakage.
- Reproducible dataset partitioning.
- Better organization of the training and evaluation pipeline.

The preprocessing stage establishes the foundation for the classification and segmentation experiments described in the following chapters.

## 16. Chapter Summary

This chapter described the dataset and preprocessing procedures used in BrainTumorResearch.

The project uses a Figshare brain tumor MRI dataset containing 3,064 images from 233 patients across three tumor categories.

MRI images are prepared using a TensorFlow-based data loading pipeline with a target input resolution of 256 × 256 pixels.

Image values are decoded as unsigned 16-bit integers, converted to floating-point values, and normalized by dividing by 65535.

Segmentation masks are resized using nearest-neighbor interpolation to preserve their categorical values.

The dataset is divided into training, validation, and testing partitions using a patient-independent 70/15/15 strategy.

These procedures provide the prepared data required for the classification and segmentation experiments.

---

**Document:** 02 — Dataset and Preprocessing  
**Project:** BrainTumorResearch  
**Dataset:** Figshare Brain Tumor MRI Dataset  
**Image Resolution:** 256 × 256  
**Data Split:** 70% Training / 15% Validation / 15% Testing  
**Random Seed:** 42

## 17. Implementation Details Confirmed in the Dataset Loader

The source implementation in src/data/dataset_loader.py provides the following safeguards and processing details:

1. The metadata file is read from data/processed/splits.csv by default.
2. Required columns are sample_id, patient_id, label, image_path, mask_path, and split.
3. The loader rejects missing metadata values and labels outside glioma, meningioma, and pituitary.
4. It requires the split names to be exactly train, validation, and test.
5. It checks that every referenced image and mask file exists.
6. It groups metadata by patient and raises an error if a patient appears in more than one partition. This is an explicit runtime leakage check.

The image and mask pipelines use different decoding and resizing rules:

| Data | PNG decoding | Value conversion | Resize interpolation |
|---|---|---|---|
| MRI image | One channel, uint16 | float32 divided by 65535.0 | Bilinear |
| Segmentation mask | One channel, uint8 | Binary: values greater than zero become 1 | Nearest neighbor, followed by a binary threshold |

Both outputs are assigned the shape (256, 256, 1). The classification pipeline loads images and integer class labels, while the segmentation pipeline loads paired images and masks.

The loader constructor defaults to batch size 16 and seed 42. Training scripts may override these defaults; for example, the baseline U-Net training script explicitly uses batch size 4. Batch size should therefore be reported per experiment rather than assumed to be identical across the repository.


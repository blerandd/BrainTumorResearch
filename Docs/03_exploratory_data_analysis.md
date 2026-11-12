# 03 — Exploratory Data Analysis

## 1. Introduction

Exploratory Data Analysis (EDA) is the process of examining and understanding a dataset before applying machine learning or deep learning algorithms.

In the BrainTumorResearch project, EDA is used to investigate the characteristics of the brain MRI dataset and identify patterns that may influence model training and evaluation.

The analysis focuses on:

- Distribution of tumor categories.
- Number of MRI images per patient.
- Visual inspection of MRI images.
- Segmentation mask inspection.
- Tumor-region characteristics.
- Image intensity and tumor-area analysis.
- Dataset organization and consistency.

The purpose of this stage is to understand the structure and characteristics of the dataset before training classification and segmentation models.

## 2. Objectives of Exploratory Data Analysis

The main objectives of EDA are:

1. Understand the overall dataset structure.
2. Examine the distribution of tumor categories.
3. Investigate how MRI images are distributed across patients.
4. Inspect sample MRI images from different categories.
5. Examine the relationship between MRI images and segmentation masks.
6. Investigate tumor-region characteristics.
7. Identify potential dataset inconsistencies.
8. Produce visualizations that support the research.

EDA provides useful information for designing the preprocessing pipeline and selecting appropriate evaluation strategies.

## 3. Dataset Class Distribution

The dataset contains three brain tumor categories:

- Glioma
- Meningioma
- Pituitary tumor

The distribution is:

| Tumor Category | Number of Images |
|---|---:|
| Glioma | 1,426 |
| Meningioma | 708 |
| Pituitary tumor | 930 |
| **Total** | **3,064** |

### 3.1 Interpretation

The glioma category contains the largest number of images, with 1,426 samples.

The meningioma category contains the smallest number of images, with 708 samples.

The pituitary tumor category contains 930 images.

This shows that the dataset is not perfectly balanced.

Class distribution is important because a model trained on an imbalanced dataset may perform differently across categories.

For this reason, classification evaluation should consider more than overall accuracy.

Metrics such as precision, recall, F1-score, and the confusion matrix provide additional information about performance for individual classes.

### 3.2 Class Distribution Visualization

A bar chart can be used to compare the number of MRI images in each category.

A second useful visualization is a pie chart showing the percentage represented by each tumor category.

These visualizations make it easier to identify differences in class representation.

**Photo location:**

`photos/dataset/`

## 4. Patient-Level Image Distribution

The dataset contains 3,064 MRI image slices from 233 patients.

A single patient may contribute multiple MRI images.

Therefore, examining the number of images per patient is important for understanding the dataset structure.

### 4.1 Images Per Patient

The project includes analysis of the number of MRI images associated with each patient.

This analysis helps identify whether images are distributed evenly or whether some patients contribute significantly more images than others.

A histogram can be used to visualize the number of images associated with individual patients.

### 4.2 Why Patient Distribution Matters

Patient distribution is important because individual MRI slices from the same patient are not completely independent observations.

Images belonging to one patient may share anatomical characteristics, acquisition properties, and tumor-related patterns.

If images from the same patient are distributed across training and testing partitions, evaluation may become overly optimistic.

For this reason, the project uses patient-independent dataset splitting.

The patient-level distribution analysis supports this decision.

**Photo location:**

`photos/dataset/`

## 5. MRI Image Inspection

Visual inspection is an important part of EDA because numerical summaries alone cannot fully describe the characteristics of medical images.

The project includes visualization scripts for examining sample MRI images from the dataset.

The inspection process considers:

- Image appearance.
- Tumor location.
- Tumor shape.
- Intensity patterns.
- Differences between tumor categories.
- Variations between patients.

MRI images from different categories may contain different visual characteristics.

However, these differences can be subtle, and visual inspection alone is not sufficient for reliable classification.

Deep learning models are used to learn patterns from the image data during training.

### 5.1 Sample MRI Images

Representative MRI examples can be selected from each tumor category.

Displaying samples side by side allows visual comparison between glioma, meningioma, and pituitary tumor images.

These examples also help demonstrate the type of input used by the classification models.

**Photo location:**

`photos/dataset/`

## 6. Segmentation Mask Inspection

In addition to MRI images, the dataset includes segmentation information used in the segmentation experiments.

A segmentation mask identifies the tumor region within an MRI image.

Visual inspection of the masks helps verify that the image and mask correspond correctly.

The analysis focuses on:

- Tumor-region location.
- Mask dimensions.
- Background and foreground regions.
- Correspondence between MRI images and masks.
- Visual differences between tumor regions.

### 6.1 MRI and Mask Comparison

An effective visualization displays three elements:

1. Original MRI image.
2. Ground-truth segmentation mask.
3. MRI image with the ground-truth mask overlaid.

This allows the tumor region to be examined in relation to the original MRI image.

It also provides a visual reference for understanding the segmentation task.

**Photo location:**

`photos/dataset/`

## 7. Tumor Area Analysis

The project includes analysis of tumor-region characteristics.

One important measurement is the area occupied by the tumor in the segmentation mask.

For a binary segmentation mask, tumor area can be calculated by counting the pixels assigned to the tumor region.

\[
A_{tumor} = \sum_{i=1}^{H}\sum_{j=1}^{W} M(i,j)
\]

Where:

- \(A_{tumor}\) represents the number of tumor pixels.
- \(M(i,j)\) represents the binary mask value at pixel location \((i,j)\).
- \(H\) represents image height.
- \(W\) represents image width.

When the tumor pixels are represented by 1 and the background by 0, the sum gives the total number of tumor pixels.

### 7.1 Tumor Area Distribution

Tumor area can be examined across different images and patients.

A histogram can show the distribution of tumor areas.

A boxplot can help identify differences in tumor-area distributions and potential outliers.

### 7.2 Tumor Area and Patient Analysis

The project also investigates tumor intensity and morphology in relation to patient-level information.

This analysis helps explore whether tumor characteristics vary across patients and whether those characteristics may be relevant when interpreting segmentation performance.

**Photo location:**

`photos/dataset/`

## 8. Image Intensity Analysis

MRI images contain pixel-intensity information that represents differences in image appearance.

The project includes inspection of image intensity characteristics.

Intensity analysis can help understand:

- Pixel-value distributions.
- Differences between image regions.
- Tumor and background intensity patterns.
- Variations between MRI images.

The original image values and their normalized representations are considered as part of the preprocessing pipeline.

The normalization procedure is explained in:

`02_dataset_and_preprocessing.md`

## 9. Dataset Visualization Scripts

The project contains several scripts used for dataset exploration and visualization.

Relevant files include:

| Script | Purpose |
|---|---|
| `src/data/analyze_dataset.py` | Examines dataset distribution and patient-level information. |
| `src/data/explore_dataset.py` | Supports dataset inspection and exploration. |
| `src/data/visualize_dataset.py` | Generates visualizations of dataset examples. |
| `src/data/visualize_loaded_data.py` | Examines data after loading through the data pipeline. |
| `src/data/analyze_tumor_intensity_patient.py` | Investigates tumor intensity and patient-level characteristics. |
| `src/data/inspect_cvind.py` | Inspects dataset-related information used in the project. |
| `src/data/analyze_cvind.py` | Performs additional dataset analysis. |

These scripts provide the basis for understanding the dataset before model development.

## 10. Dataset Quality and Consistency

Dataset quality is important because inconsistencies can affect model training and evaluation.

The exploratory analysis helps investigate:

- Missing or unreadable image files.
- Incorrect image and mask correspondence.
- Unexpected image dimensions.
- Differences in patient-level image counts.
- Unexpected mask values.
- Dataset organization problems.

The preprocessing pipeline also performs checks to ensure that images and segmentation masks can be loaded and prepared correctly.

These checks help reduce problems during model training.

## 11. Relationship Between EDA and Model Development

EDA provides information that influences later stages of the research.

For example:

- Class distribution helps interpret classification performance.
- Patient distribution motivates patient-independent data splitting.
- MRI inspection helps understand the input data.
- Mask inspection helps verify segmentation targets.
- Tumor-area analysis provides additional information about segmentation examples.
- Intensity analysis supports understanding of image preprocessing.

Therefore, EDA is not simply a visualization step. It contributes to the design and interpretation of the complete experimental pipeline.

## 12. Main Findings from Dataset Exploration

The initial dataset exploration establishes the following characteristics:

- The dataset contains 3,064 MRI image slices.
- The images originate from 233 patients.
- The dataset contains three tumor categories.
- Glioma has the largest number of images.
- Meningioma has the smallest number of images.
- Patients contribute multiple MRI slices.
- Segmentation masks provide pixel-level tumor-region information.
- Patient-level organization must be considered when splitting the dataset.

These characteristics influence the preprocessing, training, and evaluation strategies used in the research.

## 13. Chapter Summary

This chapter described the exploratory data analysis performed in the BrainTumorResearch project.

The analysis examines class distribution, patient-level image distribution, MRI image appearance, segmentation masks, tumor-area characteristics, and image intensity patterns.

The project includes dedicated scripts for dataset inspection, analysis, and visualization.

The findings from EDA support the use of patient-independent dataset splitting and provide a better understanding of the data used for classification and segmentation.

The next chapter focuses on the classification models developed and evaluated in the project.

---

**Document:** 03 — Exploratory Data Analysis  
**Project:** BrainTumorResearch  
**Dataset:** Figshare Brain Tumor MRI Dataset  
**Total Images:** 3,064  
**Total Patients:** 233  
**Tumor Categories:** Glioma, Meningioma, Pituitary Tumor
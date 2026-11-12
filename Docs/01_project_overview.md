# 01 — Project Overview

## 1. Introduction

Brain tumors are abnormal growths of cells within the brain. Their identification and analysis are important tasks in medical imaging because different tumor types can have different visual characteristics, locations, shapes, and structural patterns.

Magnetic Resonance Imaging (MRI) is widely used to examine brain structures and identify abnormalities. However, analyzing MRI images can be a complex task, particularly when large numbers of images must be examined.

Deep learning provides techniques that can automatically learn patterns from medical images. Convolutional Neural Networks (CNNs), in particular, are useful for image analysis because they can learn spatial features such as edges, textures, shapes, and more complex structures.

This project, **BrainTumorResearch**, investigates the application of deep learning techniques to brain tumor MRI image classification and segmentation.

The research focuses on developing, training, and evaluating deep learning models that can classify MRI images into tumor categories and identify tumor regions through image segmentation.

In addition, the project investigates model explainability, patient-level evaluation, and statistical comparisons between segmentation experiments.

## 2. Problem Definition

The main problem addressed in this research is how deep learning models can be used to analyze brain MRI images and identify relevant tumor information.

The project considers two related but different computer vision tasks.

### 2.1 Brain Tumor Classification

Classification is the task of assigning an MRI image to one of the predefined categories.

The classification component focuses on three tumor categories:

- **Glioma**
- **Meningioma**
- **Pituitary tumor**

The objective is to train models that can learn distinguishing visual patterns from MRI images and predict the corresponding tumor category.

Different classification architectures are investigated to compare their ability to recognize these categories.

### 2.2 Brain Tumor Segmentation

Segmentation is the process of identifying the specific pixels that belong to a tumor region.

Unlike classification, which produces a category for an image, segmentation produces a pixel-level prediction mask.

The segmentation component uses a U-Net-based architecture to predict tumor regions in MRI images.

The predicted segmentation masks are compared with the corresponding ground-truth masks to evaluate how accurately the models identify tumor boundaries and regions.

## 3. Research Motivation

The motivation behind this project is to investigate how deep learning can support the analysis of medical images through automated classification and segmentation.

Classification and segmentation provide complementary information.

Classification identifies the predicted tumor category, while segmentation provides spatial information about the location and shape of the tumor region.

Combining these tasks provides a broader perspective on MRI image analysis.

Another motivation is to investigate how different model architectures and training strategies influence performance.

In particular, the research examines whether data augmentation can improve segmentation performance and whether improvements observed at the image level are also reflected in patient-level evaluation.

The project also considers explainability and statistical analysis to provide a more complete understanding of model behavior rather than relying only on a single accuracy or segmentation score.

## 4. Research Objectives

The main objective is to investigate and evaluate deep learning approaches for brain tumor MRI classification and segmentation.

The specific objectives are:

1. **Dataset exploration:** Examine the MRI dataset, its organization, class distribution, and patient distribution.

2. **Data preprocessing:** Prepare MRI images and segmentation masks for deep learning experiments.

3. **Patient-independent evaluation:** Organize the dataset into training, validation, and testing partitions while preventing images from the same patient from appearing in different partitions.

4. **Classification model development:** Implement and evaluate multiple classification architectures.

5. **Segmentation model development:** Implement a U-Net-based model for identifying tumor regions.

6. **Experiment comparison:** Investigate the effects of different augmentation strategies on segmentation performance.

7. **Performance evaluation:** Use appropriate classification and segmentation metrics to assess model performance.

8. **Patient-level analysis:** Examine model performance across patients rather than relying exclusively on individual MRI slices.

9. **Statistical analysis:** Compare segmentation experiments using statistical methods and confidence intervals.

10. **Explainability:** Apply Grad-CAM to visualize image regions that contribute to classification predictions.

11. **Failure analysis:** Examine incorrect classifications and segmentation errors to understand model limitations.

## 5. Research Questions

The project investigates the following research questions:

**RQ1 — Classification Performance**

How effectively can deep learning architectures classify brain MRI images into glioma, meningioma, and pituitary tumor categories?

**RQ2 — Classification Model Comparison**

How do different classification architectures compare in terms of their predictive performance?

**RQ3 — Segmentation Performance**

How accurately can a U-Net-based architecture identify tumor regions in MRI images?

**RQ4 — Data Augmentation**

How do different data augmentation strategies influence segmentation performance?

**RQ5 — Patient-Level Generalization**

Do improvements in image-level segmentation metrics also translate into improvements when performance is evaluated at the patient level?

**RQ6 — Model Explainability**

Can Grad-CAM provide useful visual explanations of the regions that influence classification predictions?

**RQ7 — Statistical Reliability**

Are the observed differences between segmentation experiments statistically supported?

## 6. Research Scope

The project focuses on deep learning-based analysis of brain MRI images.

Its scope includes the following areas:

### 6.1 Classification

- Classification of three brain tumor categories.
- Development of CNN-based classification models.
- Comparison of different classification architectures.
- Evaluation using classification performance metrics.
- Visualization of classification results.

### 6.2 Segmentation

- Development of a U-Net-based segmentation architecture.
- Prediction of tumor regions at the pixel level.
- Comparison of different training and augmentation configurations.
- Evaluation using overlap-based and boundary-related metrics.
- Visualization of predicted masks and segmentation errors.

### 6.3 Explainability and Analysis

- Grad-CAM visualization for classification.
- Patient-level performance evaluation.
- Statistical comparison of segmentation experiments.
- Analysis of failure cases and model limitations.

The project is an experimental research implementation. Its results are intended for academic investigation and do not establish that the models are suitable for clinical diagnosis or independent medical decision-making.

## 7. Dataset Overview

The project uses the publicly available Brain Tumor MRI dataset hosted on Figshare.

**Dataset source:** https://doi.org/10.6084/m9.figshare.1512427

The dataset contains:

- **3,064** MRI image slices.
- **233** patients.
- **1,426** glioma images.
- **708** meningioma images.
- **930** pituitary tumor images.

The dataset is used for both classification and segmentation experiments.

The research uses patient-independent data partitioning to reduce the possibility of data leakage. Images belonging to the same patient are kept within the same partition.

Further information about the dataset, its organization, preprocessing, and splitting methodology is provided in:

`02_dataset_and_preprocessing.md`

## 8. Research Methodology

The research follows a structured experimental workflow.

### Stage 1 — Dataset Acquisition and Exploration

The first stage involves obtaining the dataset and examining its structure.

The dataset is analyzed to understand:

- The number of images.
- The distribution of tumor categories.
- The number of images associated with each patient.
- The availability and organization of segmentation masks.

### Stage 2 — Data Preprocessing

MRI images and masks are prepared for model training.

The preprocessing pipeline includes:

- Image loading and decoding.
- Image resizing.
- Pixel-value normalization.
- Segmentation mask preparation.
- Patient-independent dataset splitting.

The implementation uses an image input size of 256 × 256 pixels for the model data pipeline.

### Stage 3 — Classification Experiments

Classification models are developed and trained using the prepared MRI images.

The project investigates different architectures, including:

- Custom CNN.
- VGG16.
- ResNet50.

Their performance is evaluated using classification metrics and visual analysis.

### Stage 4 — Segmentation Experiments

A U-Net-based architecture is implemented for tumor segmentation.

Different training configurations are investigated, including:

- Baseline U-Net.
- U-Net with flip augmentation.
- U-Net with flip and rotation augmentation.

The purpose is to examine how augmentation strategies influence the model's ability to identify tumor regions.

### Stage 5 — Model Evaluation

The trained models are evaluated using appropriate metrics.

Classification evaluation considers measures such as accuracy, precision, recall, F1-score, and ROC-AUC.

Segmentation evaluation considers measures such as Dice coefficient, Intersection over Union (IoU), and boundary-related metrics.

### Stage 6 — Patient-Level and Statistical Analysis

The research investigates performance at the patient level to provide an additional perspective beyond individual image slices.

Statistical comparisons are also performed to examine differences between segmentation experiments.

### Stage 7 — Explainability and Failure Analysis

Grad-CAM is used to visualize regions influencing classification predictions.

Failure analysis is performed to investigate examples where models produce incorrect classifications or inaccurate segmentation masks.

### Stage 8 — Results and Discussion

The final stage brings together the experimental results and discusses the performance, differences between models, limitations, and possible future improvements.

## 9. Technologies and Frameworks

The project is implemented primarily in Python.

The main technologies and libraries include:

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| TensorFlow | Deep learning framework |
| Keras | Neural network implementation and training |
| NumPy | Numerical operations |
| Pandas | Data organization and analysis |
| Scikit-learn | Classification evaluation and statistical utilities |
| Matplotlib | Visualization and plotting |
| Seaborn | Statistical visualizations |
| OpenCV | Image processing |
| Pillow | Image loading and manipulation |

The complete dependency list is maintained in the root-level `requirements.txt` file.

## 10. Expected Research Contributions

The project aims to provide an experimental comparison of deep learning techniques for brain tumor MRI analysis.

Its main contributions include:

- An implemented pipeline for brain tumor MRI classification.
- An implemented U-Net-based segmentation pipeline.
- A comparison of different classification architectures.
- An investigation of segmentation augmentation strategies.
- Patient-independent data partitioning and evaluation.
- Patient-level segmentation analysis.
- Statistical comparisons between segmentation experiments.
- Grad-CAM visualizations for classification explainability.
- Failure-case analysis to investigate model limitations.

These contributions are intended to support a more detailed understanding of the strengths and limitations of the investigated approaches.

## 11. Organization of the Research Repository

The repository is organized into implementation, experimental results, saved models, and documentation.

The main components include:

- `src/` — Source code for dataset processing, model architectures, training, and evaluation.
- `models/` — Saved model checkpoints.
- `results/` — Experimental outputs, metrics, graphs, and visualizations.
- `docs/` — Technical and research documentation.
- `requirements.txt` — Python dependencies.
- `README.md` — Main repository introduction.

The dataset itself is not stored inside the documentation directory.

## 12. Chapter Summary

This chapter introduced the BrainTumorResearch project, its research problem, motivation, objectives, research questions, and experimental scope.

The research investigates brain tumor MRI classification and segmentation using deep learning techniques.

It also examines model comparison, augmentation strategies, patient-level evaluation, statistical analysis, explainability, and failure cases.

The following chapter describes the dataset in more detail and explains the preprocessing procedures used to prepare MRI images and segmentation masks for the experiments.

---

**Document:** 01 — Project Overview  
**Project:** BrainTumorResearch  
**Research Areas:** Deep Learning, Medical Image Classification, Medical Image Segmentation, Explainable AI
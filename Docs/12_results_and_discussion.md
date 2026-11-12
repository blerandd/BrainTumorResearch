# 12 — Results and Discussion

## 1. Introduction

This chapter presents and discusses the experimental results obtained in BrainTumorResearch.

The project investigates deep learning techniques for brain tumor MRI analysis through two main tasks:

1. Brain tumor classification.
2. Brain tumor segmentation.

The classification component investigates how accurately different CNN architectures identify tumor categories.

The segmentation component investigates how accurately U-Net models identify tumor regions at the pixel level.

In addition to quantitative evaluation, the research includes patient-level analysis, statistical comparisons, explainability, and failure case investigation.

The objective of this chapter is to bring these findings together and interpret their significance.

## 2. Overview of the Experiments

The research follows a structured experimental workflow.

The main stages include:

1. Dataset preparation and preprocessing.
2. Exploratory data analysis.
3. Classification model development.
4. Segmentation model development.
5. Model training and experimentation.
6. Quantitative evaluation.
7. Patient-level analysis.
8. Statistical analysis.
9. Grad-CAM explainability.
10. Failure case analysis.

The experiments use brain MRI images from three tumor categories:

- Glioma.
- Meningioma.
- Pituitary tumor.

The dataset contains 3,064 MRI images from 233 patients.

The dataset is divided into training, validation, and testing partitions using a patient-independent splitting strategy.

The split proportions are:

- 70% training.
- 15% validation.
- 15% testing.

The dataset preparation and experimental methodology are described in the earlier chapters.

## 3. Classification Results

### 3.1 Classification Objective

The classification component aims to predict the tumor category represented in an MRI image.

The model receives an MRI image as input and produces a prediction corresponding to one of three classes:

- Glioma.
- Meningioma.
- Pituitary tumor.

The research investigates CNN-based architectures, including:

- Custom CNN.
- VGG16.
- ResNet50.

These architectures provide different approaches to learning visual features from MRI images.

### 3.2 Recorded Classification Performance

The recorded classification evaluation includes the following results:

| Metric | Result |
|---|---:|
| Test images | 489 |
| Accuracy | 0.8466 |
| Macro ROC-AUC | 0.9598 |

The classification accuracy is:

\[
Accuracy = 0.8466
\]

Expressed as a percentage:

\[
Accuracy = 84.66\%
\]

This means that approximately 84.66% of the evaluated test images were classified correctly.

The macro ROC-AUC is:

\[
Macro\ ROC\text{-}AUC = 0.9598
\]

This indicates strong class-discrimination performance under the reported evaluation.

### 3.3 Interpretation of Classification Results

The accuracy result indicates that the evaluated classification model correctly identifies a substantial proportion of the test images.

The macro ROC-AUC provides additional information about the model's ability to distinguish between the tumor categories across classification thresholds.

Accuracy and ROC-AUC measure different aspects of classification performance.

A model can achieve a strong ROC-AUC while still making incorrect predictions at its selected classification decision thresholds.

Therefore, both metrics should be considered alongside the confusion matrix, precision, recall, and F1-score.

**Important:** The recorded values above should be associated with the exact model checkpoint and evaluation configuration that produced them. They should not be presented as the combined results of all three architectures unless the corresponding experiments confirm this.

## 4. Segmentation Results

### 4.1 Segmentation Objective

The segmentation component aims to identify tumor regions within brain MRI images.

Unlike classification, which predicts one category for an entire image, segmentation produces a pixel-level prediction.

The project uses U-Net architectures for this task.

The predicted segmentation mask is compared with the corresponding ground-truth mask.

The Dice coefficient is used to measure the overlap between the prediction and reference mask.

### 4.2 U-Net Experimental Configurations

Three segmentation configurations were investigated:

1. Baseline U-Net.
2. Flip-only U-Net.
3. Flip and rotation U-Net.

The experiments investigate whether data augmentation can improve segmentation performance.

The baseline model provides a reference point.

The flip-only configuration introduces flipping augmentation.

The flip-and-rotation configuration combines flipping with rotation augmentation.

### 4.3 Recorded Segmentation Results

The recorded image-level mean Dice scores are:

| Model Configuration | Image-Level Mean Dice |
|---|---:|
| Baseline U-Net | 0.7016 |
| Flip-only U-Net | 0.7259 |
| Flip and rotation U-Net | 0.7281 |

The results show that both augmented configurations achieved higher image-level mean Dice scores than the baseline.

The difference between baseline and flip-only is:

\[
0.7259 - 0.7016 = 0.0243
\]

The difference between baseline and flip and rotation is:

\[
0.7281 - 0.7016 = 0.0265
\]

The difference between flip and rotation and flip-only is:

\[
0.7281 - 0.7259 = 0.0022
\]

These calculations describe numerical differences between the recorded image-level mean Dice scores.

They do not independently establish statistical significance.

### 4.4 Interpretation of Segmentation Results

The flip-only configuration improved the image-level mean Dice score compared with the baseline.

The flip-and-rotation configuration achieved the highest recorded image-level mean Dice score.

However, the numerical improvement between flip-only and flip-and-rotation was relatively small.

For this reason, patient-level statistical comparisons are necessary to investigate whether the differences represent consistent improvements across patients.

## 5. Patient-Level Results

Image-level evaluation provides an overall summary of segmentation performance, but it does not fully describe how consistently a model performs across different patients.

The project therefore includes patient-level analysis.

The recorded paired statistical comparison includes:

**35 patients.**

The main comparison between flip-only U-Net and baseline U-Net produced the following results:

| Measurement | Result |
|---|---:|
| Number of patients | 35 |
| Mean paired Dice difference | +0.0403 |
| Confidence interval | [0.0103, 0.0761] |
| Holm-adjusted p-value | 0.0019 |

The difference is calculated as:

\[
Dice_{flip-only} - Dice_{baseline}
\]

The positive mean difference indicates that flip-only U-Net achieved a higher average patient-level Dice score than the baseline configuration.

The confidence interval is entirely above zero.

This supports a positive difference under the reported analysis.

The adjusted p-value is also below the commonly used significance threshold of 0.05.

Therefore, the flip-only versus baseline comparison was statistically significant after the reported Holm correction.

## 6. Statistical Comparison of Segmentation Models

The project investigates three pairwise comparisons:

1. Flip-only versus baseline.
2. Flip and rotation versus baseline.
3. Flip and rotation versus flip-only.

The recorded statistical analysis found a significant improvement for flip-only compared with baseline.

However, the comparisons involving flip and rotation did not establish statistically significant differences.

This is an important distinction.

Although flip and rotation achieved the highest image-level mean Dice score, the patient-level statistical analysis did not establish that this configuration performed significantly better than the other configurations.

A higher numerical average does not automatically demonstrate a statistically reliable improvement.

The findings should therefore be interpreted using both image-level and patient-level results.

## 7. Classification and Segmentation: Different Objectives

Classification and segmentation are related but different tasks.

Classification predicts the tumor category for an entire MRI image.

Segmentation identifies the tumor region within the image.

Their results should not be compared directly because they measure different objectives.

| Aspect | Classification | Segmentation |
|---|---|---|
| Main objective | Identify tumor category | Identify tumor region |
| Output | Class prediction | Pixel-level mask |
| Main model family | CNN architectures | U-Net |
| Main evaluation | Accuracy, precision, recall, F1, ROC-AUC | Dice, IoU, boundary metrics |
| Main question | What tumor category is represented? | Where is the tumor located? |

Together, these tasks provide complementary information about brain MRI analysis.

## 8. Explainability Results

The project also investigates classification explainability using Grad-CAM.

Grad-CAM generates heatmaps that highlight image regions contributing to a selected class prediction.

The purpose is to investigate whether the classification model focuses on potentially relevant regions of the MRI image.

Explainability complements the numerical classification results.

For example, an image may be classified correctly, but the Grad-CAM visualization can provide additional information about which regions contributed to the prediction.

Similarly, incorrect predictions can be investigated using their corresponding heatmaps.

However, Grad-CAM does not prove that a model has learned medically correct features.

It also does not produce an exact tumor segmentation mask.

The explainability outputs are stored under:

`results/explainability/`

## 9. Failure Case Analysis

Average performance metrics do not describe every prediction made by a model.

For this reason, the research also investigates failure cases.

Classification failure analysis focuses on incorrect tumor category predictions.

Segmentation failure analysis focuses on inaccurate predicted masks.

Possible segmentation errors include:

- False positive regions.
- False negative regions.
- Inaccurate tumor boundaries.
- Fragmented predictions.
- Incomplete tumor coverage.

The project also contains patient-level failure analysis outputs, including:

`results/segmentation/patient_level/misclassified_patient_analysis.csv`

These results support the investigation of difficult cases.

Failure analysis is important because it helps identify situations where the models perform less reliably.

It also provides context for interpreting the average classification and segmentation metrics.

## 10. Main Findings

The experiments provide several important findings.

### Finding 1 — Classification Performance

The recorded classification evaluation achieved:

- Accuracy: 0.8466.
- Macro ROC-AUC: 0.9598.
- Test images: 489.

These results indicate that the evaluated classification model achieved strong performance under the reported test evaluation.

### Finding 2 — Data Augmentation Improved Image-Level Segmentation Scores

Both augmented U-Net configurations achieved higher image-level mean Dice scores than the baseline.

The flip-only configuration achieved 0.7259 compared with 0.7016 for the baseline.

The flip-and-rotation configuration achieved 0.7281.

### Finding 3 — Flip-Only Improvement Was Statistically Significant

The paired patient-level comparison between flip-only and baseline U-Net showed:

- Mean difference: +0.0403.
- Confidence interval: [0.0103, 0.0761].
- Holm-adjusted p-value: 0.0019.

This comparison provides statistical evidence of improved patient-level Dice performance for flip-only U-Net.

### Finding 4 — The Highest Image-Level Score Was Not Necessarily Statistically Superior

Although flip and rotation achieved the highest image-level mean Dice score, its patient-level comparisons did not establish statistically significant differences.

This demonstrates why numerical performance and statistical significance must be interpreted separately.

### Finding 5 — Explainability Adds a Different Perspective

Grad-CAM provides visual information about the image regions contributing to classification predictions.

It complements quantitative metrics but does not independently establish model correctness or clinical reliability.

### Finding 6 — Failure Analysis Helps Identify Model Limitations

Investigating incorrect classifications and inaccurate segmentation masks provides information that average metrics cannot communicate.

This is important for understanding where further model improvements may be needed.

## 11. Discussion

The results demonstrate the value of evaluating brain tumor classification and segmentation using multiple complementary methods.

The classification experiment provides evidence about the model's ability to distinguish between three tumor categories.

The segmentation experiments investigate how data augmentation influences tumor localization.

The image-level Dice results show that both augmentation strategies improved the average segmentation score compared with the baseline.

However, the patient-level statistical analysis provides a more detailed perspective.

The flip-only configuration showed a statistically significant improvement over the baseline across the paired patient-level comparison.

The flip-and-rotation configuration achieved a slightly higher image-level mean Dice score, but its patient-level comparisons did not establish statistically significant differences.

This suggests that the highest numerical score should not automatically be interpreted as evidence of the most reliable improvement.

The findings also demonstrate the importance of examining individual patients and difficult cases rather than relying exclusively on aggregate performance.

Finally, Grad-CAM and failure case analysis provide additional information about model behavior and limitations.

Together, these approaches create a more complete experimental evaluation.

## 12. Research Implications

The findings have several implications for the development of deep learning models for brain tumor MRI analysis.

### 12.1 Data Augmentation Should Be Evaluated Carefully

Data augmentation can improve model performance, but different augmentation strategies may produce different results.

The effect of an augmentation strategy should be examined using both quantitative and patient-level evaluation.

### 12.2 Patient-Level Evaluation Is Important

When datasets contain multiple images from individual patients, image-level averages alone may not fully represent model performance across individuals.

Patient-level comparisons provide additional information about consistency.

### 12.3 Statistical Evidence Matters

Model selection should not depend exclusively on the highest average score.

Confidence intervals and adjusted p-values provide additional evidence when comparing experimental configurations.

### 12.4 Explainability Supports Model Investigation

Grad-CAM can help researchers investigate the image regions contributing to classification predictions.

However, visual explanations should not be treated as proof of clinical reliability.

### 12.5 Failure Analysis Supports Further Development

Difficult cases can reveal weaknesses that are not immediately visible in average performance metrics.

These cases can help guide future experiments and improvements.

## 13. Limitations of the Experimental Results

The findings should be interpreted within the limitations of the research.

### 13.1 Dataset Size and Diversity

The experiments use a dataset containing 3,064 MRI images from 233 patients.

A larger and more diverse dataset could provide additional evidence about generalization.

### 13.2 Patient-Level Sample Size

The recorded paired patient-level statistical comparisons include 35 patients.

The findings should therefore be interpreted within the scope of this evaluated sample.

### 13.3 Dataset Dependence

The results reflect the dataset, preprocessing pipeline, model configurations, and experimental conditions used in this project.

They do not guarantee equivalent performance on independent external datasets.

### 13.4 Classification and Segmentation Metrics

Classification and segmentation measure different objectives.

Their numerical results cannot be interpreted as directly comparable measures of overall model quality.

### 13.5 Explainability Limitations

Grad-CAM provides class-specific visual explanations, but it cannot establish that the model's reasoning is medically correct.

### 13.6 Clinical Validation

The research is an experimental deep learning study.

The results do not establish that the models are ready for clinical diagnosis or deployment.

Additional validation and clinical assessment would be necessary.

## 14. Chapter Summary

This chapter brought together the main experimental results from BrainTumorResearch.

The classification evaluation achieved an accuracy of 0.8466 and a macro ROC-AUC of 0.9598 on 489 test images.

For segmentation, the baseline, flip-only, and flip-and-rotation configurations achieved image-level mean Dice scores of 0.7016, 0.7259, and 0.7281, respectively.

The paired patient-level comparison showed a statistically significant improvement for flip-only U-Net over the baseline, with a mean difference of +0.0403, a confidence interval of [0.0103, 0.0761], and a Holm-adjusted p-value of 0.0019.

The comparisons involving flip and rotation did not establish statistically significant differences.

Grad-CAM and failure case analysis provided additional perspectives on model behavior and limitations.

Overall, the research demonstrates the importance of combining quantitative evaluation, patient-level analysis, statistical testing, explainability, and failure case investigation when studying deep learning models for brain tumor MRI analysis.

The following chapter focuses on reproducibility and the software environment, documenting the tools, libraries, project structure, and procedures needed to understand and reproduce the experiments.

---

**Document:** 12 — Results and Discussion  
**Project:** BrainTumorResearch  
**Main Tasks:** Brain Tumor Classification and Segmentation  
**Classification Accuracy:** 0.8466  
**Classification Macro ROC-AUC:** 0.9598  
**Baseline U-Net Mean Dice:** 0.7016  
**Flip-Only U-Net Mean Dice:** 0.7259  
**Flip-and-Rotation U-Net Mean Dice:** 0.7281  
**Patient-Level Comparison:** 35 patients  
**Main Statistical Finding:** Flip-only U-Net outperformed baseline U-Net in the reported paired patient-level comparison.
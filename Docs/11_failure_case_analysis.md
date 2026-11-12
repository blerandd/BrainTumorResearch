# 11 — Failure Case Analysis

## 1. Introduction

Failure case analysis is an important part of evaluating deep learning models.

A model's overall accuracy or average segmentation score provides useful information about its performance. However, these metrics do not explain every mistake made by the model.

Two models may achieve similar average scores while making very different types of errors.

For this reason, it is important to investigate individual cases where the model produces incorrect predictions or inaccurate segmentation masks.

In BrainTumorResearch, failure case analysis focuses on two main tasks:

1. Brain tumor classification.
2. Brain tumor segmentation.

Classification analysis investigates images assigned to the wrong tumor category.

Segmentation analysis investigates MRI images where the predicted tumor mask does not accurately match the ground-truth mask.

The objective is to understand where the models struggle, identify possible sources of errors, and evaluate the limitations of the experimental approach.

## 2. Why Failure Case Analysis Matters

Evaluating a model only through average performance metrics can hide important weaknesses.

For example, a classification model may achieve good overall accuracy but still struggle to distinguish between two tumor categories.

Similarly, a segmentation model may achieve a reasonable mean Dice score while producing inaccurate masks for particular patients.

Failure case analysis helps answer questions such as:

- Which images are incorrectly classified?
- Which tumor categories are more difficult to distinguish?
- Which MRI images produce poor segmentation results?
- Are errors concentrated in particular patients?
- Does the model miss parts of the tumor?
- Does the model predict regions outside the actual tumor?
- Are errors associated with difficult image characteristics?

Understanding these cases provides a more complete evaluation of model behavior.

## 3. Classification Failure Analysis

Classification failure analysis focuses on MRI images for which the predicted tumor category does not match the ground-truth label.

The classification model predicts one of three categories:

- Glioma.
- Meningioma.
- Pituitary tumor.

A classification error occurs when:

\[
Predicted\ Class \neq True\ Class
\]

For example:

| True Class | Predicted Class | Result |
|---|---|---|
| Glioma | Glioma | Correct |
| Glioma | Meningioma | Incorrect |
| Meningioma | Pituitary | Incorrect |
| Pituitary | Pituitary | Correct |

Incorrect predictions should be examined individually to understand the model's behavior.

## 4. Types of Classification Errors

Several types of classification errors may occur.

### 4.1 False Positive Predictions

A false positive occurs when a model predicts a particular class for an image that does not belong to that class.

For example, an image belonging to the meningioma category may incorrectly be predicted as glioma.

### 4.2 False Negative Predictions

A false negative occurs when an image belonging to a particular class is incorrectly assigned to another class.

For example, a glioma image may be predicted as meningioma.

### 4.3 Confusion Between Tumor Categories

Some MRI images may contain visual characteristics that make classification more difficult.

A model may confuse two tumor categories because of similarities in image appearance.

The confusion matrix helps identify which class pairs are most frequently confused.

However, the confusion matrix alone does not explain why an individual prediction was incorrect.

Incorrect examples should therefore be inspected alongside their original MRI images and, where available, Grad-CAM visualizations.

## 5. Investigating Incorrect Classification Predictions

A useful failure analysis should include the following information for each selected image:

- Image identifier.
- Ground-truth class.
- Predicted class.
- Prediction confidence, if available.
- Original MRI image.
- Grad-CAM visualization, if available.

This information allows the researcher to compare the actual category with the model's prediction.

Grad-CAM can provide additional information about which image regions contributed to the predicted class score.

For example, if an image belongs to the glioma category but is classified as meningioma, its Grad-CAM visualization can help investigate which regions influenced that prediction.

However, a heatmap does not independently establish the reason for the classification error.

The original image, true label, prediction, and explanation should be considered together.

## 6. Segmentation Failure Analysis

Segmentation failure analysis investigates cases where the predicted tumor mask does not accurately match the ground-truth mask.

The U-Net model generates a pixel-level segmentation prediction.

The predicted mask is compared with the reference mask to evaluate how accurately the model identifies the tumor region.

Segmentation errors can occur even when the predicted mask looks visually similar to the reference mask.

For example, a model may:

- Miss part of the tumor.
- Predict additional regions outside the tumor.
- Produce an incomplete tumor boundary.
- Fail to identify small tumor regions.
- Generate irregular or fragmented predictions.

These errors can reduce the Dice score and other segmentation metrics.

## 7. Types of Segmentation Failures

### 7.1 False Negative Regions

False negative pixels occur when the ground-truth mask contains tumor pixels that the model fails to predict.

In other words, the model misses part of the actual tumor region.

Possible visual characteristics include:

- Missing tumor sections.
- Incomplete tumor coverage.
- Small tumor regions that are not detected.

False negative regions reduce segmentation overlap.

### 7.2 False Positive Regions

False positive pixels occur when the model predicts tumor pixels outside the ground-truth tumor region.

This means that the model identifies regions as tumor even though they are not included in the reference mask.

Possible visual characteristics include:

- Excessive predicted tumor areas.
- Incorrectly highlighted surrounding tissue.
- Additional regions outside the reference boundary.

False positive regions also reduce segmentation overlap.

### 7.3 Boundary Errors

Boundary errors occur when the predicted tumor contour does not match the ground-truth contour.

The model may identify most of the tumor correctly but still produce an inaccurate boundary.

This is particularly important because Dice alone does not fully describe the geometric accuracy of a segmentation boundary.

The project also includes boundary-related evaluation using a 2-pixel tolerance.

### 7.4 Fragmented Predictions

A segmentation model may produce disconnected predicted regions instead of one coherent tumor region.

Such predictions may indicate difficulty identifying the complete spatial structure of the tumor.

These cases should be inspected visually rather than judged only by the average Dice score.

## 8. Visual Comparison of Segmentation Failures

A useful way to investigate segmentation errors is to display three images together:

1. Original MRI image.
2. Ground-truth segmentation mask.
3. Predicted segmentation mask.

This comparison makes it easier to identify missing regions, additional predicted regions, and boundary differences.

A fourth visualization can show an overlay of the prediction and reference mask.

When examining these images, the following questions are useful:

- Does the predicted mask cover the complete tumor?
- Are additional regions incorrectly classified as tumor?
- Does the predicted boundary follow the reference boundary?
- Are small tumor regions missed?
- Are there large differences between the prediction and reference mask?

The project contains segmentation visualization outputs that can support this analysis.

## 9. Patient-Level Failure Analysis

Failure analysis becomes especially important when multiple MRI images belong to the same patient.

A model may perform well on most patients but struggle with particular individuals.

Patient-level analysis helps identify whether errors are distributed across the dataset or concentrated in a smaller number of patients.

The project includes patient-level segmentation evaluation.

Relevant outputs include:

`results/segmentation/patient_level/misclassified_patient_analysis.csv`

This file supports investigation of patient-level cases identified during the analysis.

Patient-level results should be considered together with the corresponding image-level predictions.

A low patient-level score may indicate that the model struggles with several images from the same individual.

However, the specific reason must be investigated using the available MRI images and segmentation outputs.

## 10. Relationship Between Failure Cases and Dice Score

The Dice coefficient measures the overlap between the predicted tumor region and the ground-truth mask.

It is calculated as:

\[
Dice = \frac{2|P \cap G|}{|P| + |G|}
\]

Where:

- \(P\) represents the predicted tumor region.
- \(G\) represents the ground-truth tumor region.
- \(P \cap G\) represents the overlapping region.

A Dice score closer to 1 indicates stronger overlap.

A score closer to 0 indicates weaker overlap.

Failure cases help explain why individual images or patients may achieve lower Dice scores.

For example:

- Missing tumor regions increase false negatives.
- Additional predicted regions increase false positives.
- Incorrect boundaries reduce overlap.
- Fragmented predictions can reduce the agreement between masks.

Therefore, visual failure analysis complements quantitative segmentation metrics.

## 11. Comparing Failure Cases Across U-Net Experiments

The project evaluates three segmentation configurations:

1. Baseline U-Net.
2. Flip-only U-Net.
3. Flip and rotation U-Net.

Their recorded image-level mean Dice scores are:

| Model | Image-Level Mean Dice |
|---|---:|
| Baseline U-Net | 0.7016 |
| Flip-only U-Net | 0.7259 |
| Flip and rotation U-Net | 0.7281 |

These values provide an overall comparison of segmentation performance.

However, they do not show whether the same images or patients are difficult for all three models.

A more detailed failure analysis can compare predictions from the three configurations on corresponding MRI images.

This makes it possible to investigate:

- Whether augmentation improves difficult cases.
- Whether a model corrects errors made by another configuration.
- Whether particular patients remain difficult across all models.
- Whether improved average performance comes from consistent improvements or a smaller number of cases.

The recorded patient-level statistical analysis should also be considered when interpreting these comparisons.

## 12. Possible Sources of Model Errors

Several factors may contribute to classification and segmentation failures.

### 12.1 Image Characteristics

MRI images may differ in intensity, contrast, anatomical appearance, and tumor visibility.

These variations can make image interpretation more difficult for a model.

### 12.2 Tumor Characteristics

Tumors may differ in:

- Size.
- Shape.
- Location.
- Boundary appearance.
- Visual characteristics.

These differences can influence both classification and segmentation performance.

### 12.3 Dataset Representation

Some image patterns may be less represented in the training data.

A model may therefore perform less consistently on patterns that it encounters less frequently.

### 12.4 Model Limitations

A CNN learns patterns from its training data.

Its predictions depend on the quality of the input data, training procedure, architecture, and learned representations.

A model may fail to capture some complex visual characteristics.

### 12.5 Segmentation Boundary Complexity

Tumor boundaries may be irregular or difficult to distinguish from surrounding tissue.

This can make accurate pixel-level segmentation challenging.

These factors are possible explanations to investigate, not automatically established causes of every failure.

## 13. Failure Analysis and Grad-CAM

Grad-CAM can be useful when investigating classification errors.

It generates a heatmap showing regions that contribute to a selected class score.

For an incorrectly classified MRI image, the researcher can compare:

- The original MRI image.
- The true class.
- The predicted class.
- The Grad-CAM explanation.

This can help investigate whether the model emphasizes potentially relevant or irrelevant image regions.

Grad-CAM should not be used as a segmentation reference.

It explains class-specific activation patterns rather than generating an exact tumor mask.

The theory and implementation context of Grad-CAM are described in:

`10_explainability_gradcam.md`

## 14. Failure Analysis Outputs and Project Structure

The BrainTumorResearch repository contains evaluation outputs and visualizations that support failure analysis.

Relevant locations include:

`results/segmentation/`

`results/segmentation/patient_level/`

`results/explainability/`

The patient-level analysis includes:

`results/segmentation/patient_level/misclassified_patient_analysis.csv`

The evaluation outputs can be used to inspect difficult cases and compare model predictions with their corresponding ground-truth masks.

Selected examples should be copied into:

`docs/photos/failure_analysis/`

Recommended documentation images include:

1. Correct classification example.
2. Incorrect classification example.
3. Original MRI image and ground-truth segmentation.
4. Predicted segmentation with false negative regions.
5. Predicted segmentation with false positive regions.
6. Difficult patient-level segmentation example.
7. Comparison of the three U-Net configurations on the same MRI image.

Only use examples that can be verified from the actual saved project outputs.

Do not present an illustrative image as an experimental result.

## 15. Limitations of Failure Case Analysis

Failure case analysis has several limitations.

### 15.1 Selected Examples May Not Represent the Entire Dataset

A small number of visual examples cannot describe every model failure.

The selected cases should be considered alongside the complete evaluation metrics.

### 15.2 Identifying Errors Does Not Establish Their Cause

An incorrect prediction can be observed directly, but its underlying cause may not be immediately clear.

Additional experiments may be required to investigate the reason.

### 15.3 Visual Interpretation Can Be Subjective

Researchers may interpret the same segmentation output differently.

Quantitative metrics should therefore accompany visual inspection.

### 15.4 Dataset-Specific Findings

The observed failures reflect the dataset and experimental conditions used in this project.

Different datasets may contain different challenges.

## 16. Chapter Summary

This chapter described failure case analysis for the classification and segmentation components of BrainTumorResearch.

Classification failure analysis investigates incorrect tumor category predictions.

Segmentation failure analysis examines inaccurate predicted masks, including false positive regions, false negative regions, boundary errors, and fragmented predictions.

Patient-level analysis provides an additional perspective by identifying cases where model performance may be weaker for particular individuals.

The project includes patient-level evaluation outputs and saved visualizations that can support this investigation.

Failure case analysis is important because average performance metrics alone cannot explain every model error.

By examining incorrect predictions and difficult segmentation cases, researchers can better understand model limitations and identify areas for future improvement.

The following chapter focuses on the overall experimental results and discussion, bringing together the classification, segmentation, statistical, and explainability findings.
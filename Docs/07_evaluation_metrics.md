# 07 — Evaluation Metrics

## 1. Introduction

Evaluation metrics are used to measure how effectively a trained deep learning model performs its intended task.

In the BrainTumorResearch project, evaluation is performed for two main tasks:

- Brain tumor MRI classification.
- Brain tumor MRI segmentation.

These tasks require different evaluation methods.

Classification metrics measure how accurately a model assigns MRI images to the correct tumor categories.

Segmentation metrics measure how closely the predicted tumor regions match the corresponding ground-truth masks.

The project also investigates patient-level performance and boundary-related segmentation metrics to provide a more detailed evaluation.

The purpose of this chapter is to explain the metrics used to interpret model performance and compare experimental results.

## 2. Classification Evaluation

Classification is a multi-class prediction task involving three categories:

- Glioma.
- Meningioma.
- Pituitary tumor.

The model receives an MRI image and predicts a probability for each category.

The predicted category is determined by selecting the class with the highest predicted probability.

Classification performance is evaluated by comparing predicted labels with the actual labels.

The main classification metrics include:

- Accuracy.
- Precision.
- Recall.
- F1-score.
- ROC-AUC.
- Confusion matrix.

## 3. Confusion Matrix

A confusion matrix is used to compare the actual class labels with the predicted class labels.

For a three-class classification problem, the confusion matrix contains three rows and three columns.

The rows represent the actual categories, while the columns represent the predicted categories.

| Actual / Predicted | Glioma | Meningioma | Pituitary |
|---|---|---|---|
| Glioma | Correct glioma predictions | Glioma predicted as meningioma | Glioma predicted as pituitary |
| Meningioma | Meningioma predicted as glioma | Correct meningioma predictions | Meningioma predicted as pituitary |
| Pituitary | Pituitary predicted as glioma | Pituitary predicted as meningioma | Correct pituitary predictions |

The diagonal elements represent correct predictions.

The off-diagonal elements represent incorrect predictions.

### 3.1 Why the Confusion Matrix Is Important

Overall accuracy does not explain which categories a model predicts correctly or incorrectly.

A confusion matrix provides more detailed information.

For example, it can reveal whether the model frequently confuses glioma with meningioma or whether one category is more difficult to identify than the others.

This is particularly useful when the dataset contains different numbers of examples for each category.

## 4. True Positives, True Negatives, False Positives, and False Negatives

Several classification metrics are calculated using four fundamental quantities.

For an individual class in a multi-class classification problem, the remaining classes are treated as the negative category.

### 4.1 True Positive (TP)

A true positive occurs when an example belonging to a particular class is correctly predicted as that class.

### 4.2 True Negative (TN)

A true negative occurs when an example does not belong to a particular class and is correctly predicted as belonging to another class.

### 4.3 False Positive (FP)

A false positive occurs when an example belonging to another class is incorrectly predicted as the selected class.

### 4.4 False Negative (FN)

A false negative occurs when an example belongs to the selected class but the model predicts a different class.

These quantities are used to calculate accuracy, precision, recall, and F1-score.

## 5. Accuracy

Accuracy measures the proportion of all predictions that are correct.

It is calculated as:

\[
Accuracy = \frac{TP + TN}{TP + TN + FP + FN}
\]

For multi-class classification, the equivalent calculation is:

\[
Accuracy = \frac{\text{Number of correct predictions}}{\text{Total number of predictions}}
\]

### Interpretation

Accuracy provides an overall measurement of classification performance.

For example, an accuracy of 0.8466 means that approximately 84.66% of the evaluated images were classified correctly.

However, accuracy alone does not indicate whether the model performs equally well across all tumor categories.

For this reason, precision, recall, F1-score, and the confusion matrix are also important.

## 6. Precision

Precision measures how many predictions assigned to a particular class are actually correct.

It is calculated as:

\[
Precision = \frac{TP}{TP + FP}
\]

### Interpretation

A high precision indicates that relatively few examples from other categories are incorrectly assigned to the selected category.

For example, high precision for glioma means that most images predicted as glioma actually belong to the glioma category.

Precision is useful for understanding the reliability of the model's positive predictions for each class.

## 7. Recall

Recall measures how many actual examples belonging to a particular class are correctly identified.

It is calculated as:

\[
Recall = \frac{TP}{TP + FN}
\]

### Interpretation

A high recall indicates that the model identifies a large proportion of the actual examples belonging to that category.

For example, high recall for meningioma means that the model correctly identifies most of the actual meningioma images.

Recall is particularly useful for identifying categories that the model frequently misses.

## 8. F1-Score

The F1-score combines precision and recall into a single measurement.

It is calculated as:

\[
F1 = 2 \times \frac{Precision \times Recall}{Precision + Recall}
\]

### Interpretation

The F1-score is the harmonic mean of precision and recall.

A high F1-score indicates that the model achieves a good balance between identifying relevant examples and avoiding incorrect positive predictions.

It is useful when evaluating classification models on datasets where category distribution is not perfectly balanced.

## 9. Macro-Averaged Metrics

In multi-class classification, metrics can be averaged across the individual categories.

Macro averaging calculates the metric separately for each class and then gives every class equal weight.

For precision:

\[
Precision_{macro} =
\frac{P_{glioma} + P_{meningioma} + P_{pituitary}}{3}
\]

The same approach is used for macro recall and macro F1-score.

### Interpretation

Macro averaging is useful because it prevents categories with more images from dominating the final averaged metric.

Each tumor category contributes equally to the macro result, regardless of the number of examples it contains.

## 10. ROC Curve and ROC-AUC

The Receiver Operating Characteristic (ROC) curve evaluates classification performance across different decision thresholds.

It plots:

- True Positive Rate (TPR).
- False Positive Rate (FPR).

The true positive rate is equivalent to recall:

\[
TPR = \frac{TP}{TP + FN}
\]

The false positive rate is:

\[
FPR = \frac{FP}{FP + TN}
\]

The Area Under the Curve (AUC) summarizes the ROC curve.

A higher ROC-AUC generally indicates better separation between the positive and negative categories.

### 10.1 Multi-Class ROC-AUC

The BrainTumorResearch classification task contains three tumor categories.

ROC-AUC can therefore be calculated using a one-vs-rest approach.

In this approach:

1. One category is treated as the positive class.
2. The remaining categories are treated as the negative class.
3. An ROC curve is calculated for that category.
4. The process is repeated for all three categories.

The macro ROC-AUC is calculated by averaging the AUC values of the individual categories.

This gives each tumor category equal weight.

### 10.2 Recorded Classification Result

The recorded classification evaluation includes:

| Metric | Result |
|---|---:|
| Test accuracy | 0.8466 |
| Macro ROC-AUC | 0.9598 |
| Number of test images | 489 |

The recorded accuracy corresponds to approximately 84.66% correct predictions.

The macro ROC-AUC of 0.9598 indicates strong class-discrimination performance in the recorded evaluation.

These metrics measure different aspects of classification performance and should be interpreted together.

## 11. Segmentation Evaluation

Segmentation evaluation measures how closely a predicted tumor mask matches the ground-truth mask.

Unlike classification, where the output is a category, segmentation produces a spatial prediction.

The project evaluates segmentation using overlap-based and boundary-related measurements.

The main concepts include:

- True positive pixels.
- False positive pixels.
- False negative pixels.
- Dice coefficient.
- Intersection over Union.
- Boundary Dice.

## 12. Pixel-Level True Positives, False Positives, and False Negatives

For binary segmentation, each pixel is classified as either tumor or background.

The predicted mask is compared with the ground-truth mask.

### 12.1 True Positive Pixels

True positive pixels are tumor pixels correctly identified by the model.

### 12.2 False Positive Pixels

False positive pixels are background pixels incorrectly predicted as tumor.

### 12.3 False Negative Pixels

False negative pixels are tumor pixels that the model fails to identify.

These values are used to calculate segmentation metrics.

## 13. Dice Coefficient

The Dice coefficient measures the overlap between the predicted tumor region and the ground-truth tumor region.

It is calculated as:

\[
Dice = \frac{2|P \cap G|}{|P| + |G|}
\]

Where:

- \(P\) represents the predicted tumor region.
- \(G\) represents the ground-truth tumor region.
- \(P \cap G\) represents the overlapping region.

An equivalent pixel-based formulation is:

\[
Dice = \frac{2TP}{2TP + FP + FN}
\]

### Interpretation

The Dice coefficient ranges from 0 to 1.

- 1 represents perfect overlap.
- 0 represents no overlap.

A higher Dice score indicates that the predicted tumor region more closely matches the ground-truth region.

### Example

If the predicted tumor region overlaps very closely with the ground-truth mask, the Dice score will be close to 1.

If the model misses a large part of the tumor or incorrectly identifies many background pixels as tumor, the Dice score will decrease.

Dice is one of the main metrics used to compare the segmentation experiments in this project.

## 14. Intersection over Union (IoU)

Intersection over Union, also known as the Jaccard Index, measures the ratio between the intersection and union of the predicted and ground-truth regions.

It is calculated as:

\[
IoU = \frac{|P \cap G|}{|P \cup G|}
\]

An equivalent pixel-based formulation is:

\[
IoU = \frac{TP}{TP + FP + FN}
\]

### Interpretation

IoU ranges from 0 to 1.

- 1 represents perfect overlap.
- 0 represents no overlap.

A higher IoU indicates better agreement between the predicted mask and the ground-truth mask.

Dice and IoU are closely related.

Their mathematical relationship is:

\[
Dice = \frac{2IoU}{1 + IoU}
\]

IoU and Dice provide similar information, although their numerical values are not identical.

## 15. Boundary Evaluation

Overlap metrics describe how much the predicted tumor region overlaps with the ground-truth region.

However, they do not fully describe the accuracy of the predicted tumor boundary.

Two segmentation masks may have similar overlap scores while differing in the precision of their boundaries.

For this reason, the project also investigates boundary-related evaluation.

### 15.1 Boundary Dice

Boundary Dice evaluates the agreement between the boundaries of the predicted mask and the ground-truth mask.

Instead of considering only the entire tumor regions, it focuses on their boundaries.

In the project, boundary evaluation includes a tolerance of 2 pixels.

This means that a predicted boundary can be considered sufficiently close to the reference boundary when it falls within the specified tolerance.

The 2-pixel tolerance allows small spatial differences to be considered during boundary comparison.

### 15.2 Why Boundary Metrics Matter

Tumor segmentation requires accurate identification of the tumor region.

A model may produce a reasonable overall overlap score while still generating inaccurate boundaries.

Boundary-related metrics provide additional information about the spatial quality of segmentation predictions.

This is particularly relevant when evaluating segmentation models that produce slightly different tumor contours.

### 15.3 Boundary Metric Testing

The project includes dedicated boundary metric implementation and tests.

The test output records successful execution of the boundary metric unit tests.

The related evaluation implementation and outputs are included in the repository's evaluation and results directories.

## 16. Image-Level Evaluation

Image-level evaluation calculates segmentation performance separately for individual MRI images.

For each image:

1. Load the ground-truth mask.
2. Obtain the predicted segmentation mask.
3. Compare the two masks.
4. Calculate the selected segmentation metrics.
5. Store the resulting values.

The individual image scores can then be summarized using statistics such as the mean and standard deviation.

### 16.1 Mean Dice

Mean Dice represents the average Dice score across the evaluated images.

\[
MeanDice = \frac{1}{N}\sum_{i=1}^{N} Dice_i
\]

Where:

- \(N\) represents the number of evaluated images.
- \(Dice_i\) represents the Dice score for image \(i\).

Mean Dice provides an overall summary of image-level segmentation overlap.

### 16.2 Recorded Image-Level Results

The project contains the following recorded mean Dice values:

| Segmentation Model | Mean Dice |
|---|---:|
| Baseline U-Net | 0.7016 |
| Flip-only U-Net | 0.7259 |
| Flip and rotation U-Net | 0.7281 |

These values show the average image-level overlap achieved by the three segmentation configurations in the recorded evaluation.

## 17. Patient-Level Evaluation

Image-level evaluation treats each MRI slice as an individual observation.

However, multiple MRI images may originate from the same patient.

Therefore, the project also investigates performance at the patient level.

Patient-level evaluation aggregates or summarizes image-level segmentation results according to patient identity.

This provides an additional perspective on model performance across different patients.

The analysis is important because images from the same patient are not necessarily independent observations.

The patient-level evaluation is described in more detail in:

`08_patient_level_analysis.md`

## 18. Statistical Interpretation of Evaluation Metrics

Numerical differences between models do not automatically establish that one model performs significantly better than another.

For example, the recorded mean Dice scores are:

- Baseline U-Net: 0.7016.
- Flip-only U-Net: 0.7259.
- Flip and rotation U-Net: 0.7281.

Although the flip-and-rotation model has the highest recorded image-level mean Dice, the difference between it and the flip-only model is relatively small.

Statistical analysis is therefore needed to investigate whether observed differences are supported by the patient-level results.

The project performs additional comparisons using paired patient-level analysis, confidence intervals, and multiple-comparison correction.

These procedures are discussed in:

`09_statistical_analysis.md`

## 19. Why Multiple Evaluation Metrics Are Necessary

No single metric provides a complete description of deep learning model performance.

For classification:

- Accuracy describes overall correctness.
- Precision describes the reliability of positive predictions.
- Recall describes the ability to identify actual examples.
- F1-score combines precision and recall.
- ROC-AUC measures class discrimination across thresholds.
- The confusion matrix identifies patterns of classification errors.

For segmentation:

- Dice measures overlap.
- IoU measures intersection relative to union.
- Boundary Dice measures boundary agreement.
- Patient-level evaluation examines performance across individual patients.

Using several complementary metrics provides a more complete understanding of model behavior.

## 20. Evaluation Implementation and Results

The project contains evaluation scripts and saved outputs inside:

`src/evaluation/`

`results/classification/`

`results/segmentation/`

The classification results include metrics, confusion matrices, ROC curves, and prediction outputs.

The segmentation results include image-level metrics, model comparisons, boundary-related measurements, patient-level analysis, and visualizations.

These files provide the experimental evidence used to interpret the evaluation results.

## 21. Chapter Summary

This chapter described the evaluation metrics used in BrainTumorResearch.

Classification performance is measured using accuracy, precision, recall, F1-score, ROC-AUC, and confusion matrices.

Segmentation performance is measured using overlap-based metrics such as Dice and IoU, together with boundary-related evaluation.

The project also distinguishes image-level and patient-level evaluation because multiple MRI slices can originate from the same patient.

The recorded classification results include an accuracy of 0.8466 and a macro ROC-AUC of 0.9598 across 489 test images.

The recorded segmentation mean Dice scores are 0.7016 for the baseline U-Net, 0.7259 for flip-only augmentation, and 0.7281 for flip and rotation augmentation.

These measurements form the basis for the patient-level and statistical analyses discussed in the following chapters.

---

**Document:** 07 — Evaluation Metrics  
**Project:** BrainTumorResearch  
**Classification Metrics:** Accuracy, Precision, Recall, F1-score, ROC-AUC, Confusion Matrix  
**Segmentation Metrics:** Dice, IoU, Boundary Dice  
**Boundary Tolerance:** 2 pixels  
**Evaluation Levels:** Image-level and Patient-level

## Implementation Details: Standard U-Net Evaluation

The standard evaluation implementation in src/evaluation/evaluate_unet.py uses a probability threshold of 0.5 to convert model outputs into binary predictions. Ground-truth masks are also converted to binary values using a threshold of 0.5.

For each test image, the script calculates Dice, IoU, pixel precision, and pixel recall. Precision and recall use scikit-learn with zero_division=0, so a metric is set to zero rather than raising a warning when its denominator is zero.

The script then creates:
- An overall image-level summary containing mean, standard deviation, median, minimum, and maximum for each metric.
- A class summary grouped by the metadata label.
- A patient summary grouped by patient_id, where each patient's image-level scores are averaged within that patient.

**Important distinction:** the overall summary is calculated from per-image metric rows. The patient summary first averages images within each patient. These are different aggregation levels and should not be described interchangeably. The script's printed patient-level summary then averages the patient-level Dice and IoU means across patients.

The evaluation script loads models/segmentation/unet_best.keras by default, with a fallback search when that file is absent. It loads the model with the project's custom loss and Dice functions and uses compile=False.


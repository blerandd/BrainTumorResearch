# 08 — Patient-Level Analysis

## 1. Introduction

Patient-level analysis is an important component of the BrainTumorResearch project.

The dataset contains multiple MRI image slices from individual patients. Therefore, evaluating model performance only at the image level does not provide a complete understanding of how consistently a model performs across different patients.

Images from the same patient may share anatomical characteristics, tumor-related patterns, and acquisition properties.

Consequently, multiple images from one patient should not automatically be treated as independent observations.

The project investigates patient-level segmentation performance to provide an additional perspective on model generalization.

The analysis focuses on comparing the three U-Net segmentation configurations:

- Baseline U-Net.
- Flip-only U-Net.
- Flip and rotation U-Net.

The main objective is to understand whether improvements observed at the image level are also reflected when performance is examined across individual patients.

## 2. Why Patient-Level Analysis Matters

A dataset containing multiple images from the same patient introduces an important consideration during model evaluation.

For example, one patient may contribute several MRI slices, while another patient may contribute a different number of images.

If evaluation is performed only at the image level, patients contributing more images may have a greater influence on the overall result.

This means that image-level averages may not fully represent how consistently the model performs across patients.

Patient-level analysis addresses this issue by grouping or summarizing evaluation results according to patient identity.

It provides information about model behavior across individuals rather than considering every MRI slice as an entirely independent observation.

## 3. Image-Level vs Patient-Level Evaluation

Image-level and patient-level evaluation provide different perspectives on model performance.

### 3.1 Image-Level Evaluation

Image-level evaluation calculates segmentation metrics separately for each MRI image.

For example, if a patient contributes ten MRI slices, each slice receives its own Dice score.

The scores from all evaluated images can then be combined to calculate an overall mean.

The image-level mean Dice is:

\[
MeanDice_{image} =
\frac{1}{N}\sum_{i=1}^{N} Dice_i
\]

Where:

- \(N\) represents the total number of evaluated images.
- \(Dice_i\) represents the Dice score for image \(i\).

This provides an overall summary of segmentation performance across the evaluated images.

### 3.2 Patient-Level Evaluation

Patient-level evaluation groups segmentation results according to patient identity.

Instead of considering every image as an independent observation, the analysis examines performance across individual patients.

A patient-level metric can be obtained by aggregating the image-level measurements belonging to that patient.

The exact aggregation method must be consistent across the models being compared.

Patient-level evaluation helps determine whether performance improvements are distributed across patients or are mainly influenced by a subset of individuals.

### 3.3 Why Both Approaches Are Useful

Image-level evaluation describes how accurately the model segments individual MRI slices.

Patient-level evaluation provides information about how consistently the model performs across different individuals.

Using both approaches provides a more complete understanding of segmentation performance.

## 4. Patient-Independent Dataset Splitting

Patient-level analysis is closely related to the dataset splitting strategy.

The project uses patient-independent dataset splitting.

The training, validation, and testing partitions are organized so that images from the same patient do not appear in different partitions.

This prevents the model from being evaluated on images belonging to patients whose other images were already used during training.

Patient-independent splitting helps reduce the risk of data leakage and provides a more meaningful evaluation of generalization to previously unseen patients.

The dataset preparation procedure is described in:

`02_dataset_and_preprocessing.md`

## 5. Patient-Level Segmentation Comparison

The project compares three segmentation configurations:

1. Baseline U-Net.
2. Flip-only U-Net.
3. Flip and rotation U-Net.

The comparison investigates whether augmentation strategies improve segmentation performance across patients.

The analysis considers paired patient-level results.

This means that the performance of different models is compared using corresponding patient observations rather than comparing unrelated groups of patients.

A paired comparison is useful because each model is evaluated on the same patient-level evaluation set.

This allows differences in model performance to be examined while accounting for variation between patients.

## 6. Number of Evaluated Patients

The recorded patient-level statistical comparison includes:

**35 patients**

These patients provide the observations used in the paired segmentation comparisons.

The use of patient-level observations is important because multiple MRI slices from one individual are related and should not automatically be treated as independent samples in statistical analysis.

## 7. Patient-Level Dice Score

The Dice coefficient measures overlap between a predicted tumor region and its ground-truth segmentation mask.

It is calculated as:

\[
Dice = \frac{2|P \cap G|}{|P| + |G|}
\]

Where:

- \(P\) represents the predicted tumor region.
- \(G\) represents the ground-truth tumor region.

The patient-level analysis uses Dice-related performance measurements to compare the segmentation configurations across individuals.

The exact patient-level Dice aggregation procedure should be interpreted according to the evaluation implementation and the saved patient-level result files.

## 8. Paired Patient-Level Differences

A paired comparison examines the difference in performance between two models for each patient.

For a comparison between Model A and Model B:

\[
D_i = Score_{A,i} - Score_{B,i}
\]

Where:

- \(D_i\) represents the performance difference for patient \(i\).
- \(Score_{A,i}\) represents the score of Model A for patient \(i\).
- \(Score_{B,i}\) represents the score of Model B for patient \(i\).

The mean paired difference is:

\[
\bar{D} = \frac{1}{n}\sum_{i=1}^{n}D_i
\]

Where \(n\) represents the number of evaluated patients.

A positive mean difference indicates that Model A achieved a higher average score than Model B under the selected comparison.

A negative mean difference indicates the opposite.

## 9. Recorded Patient-Level Results

The project contains a paired patient-level comparison involving 35 patients.

The recorded comparison between flip-only U-Net and baseline U-Net produced the following result:

| Measurement | Result |
|---|---:|
| Number of patients | 35 |
| Mean Dice difference | +0.0403 |
| Confidence interval | [0.0103, 0.0761] |
| Holm-adjusted p-value | 0.0019 |

The difference is calculated as:

\[
Dice_{flip-only} - Dice_{baseline}
\]

The positive mean difference of 0.0403 indicates that the flip-only configuration achieved a higher average patient-level Dice score than the baseline configuration in this comparison.

The confidence interval is entirely above zero, supporting a positive difference under the reported analysis.

The Holm-adjusted p-value of 0.0019 indicates that this comparison remained statistically significant after the reported multiple-comparison correction.

## 10. Interpretation of the Flip-Only Improvement

The recorded patient-level comparison suggests that flip augmentation improved segmentation performance relative to the baseline U-Net.

The mean paired difference was:

\[
+0.0403
\]

This means that the flip-only model achieved an average patient-level Dice improvement of approximately 0.0403 compared with the baseline under the reported patient-level analysis.

Importantly, this comparison evaluates performance across patients rather than relying exclusively on individual MRI slices.

The result therefore provides additional evidence that the observed improvement was not limited to the overall image-level average.

However, the result should be interpreted within the limitations of the dataset and the experimental setup.

## 11. Comparison with Flip and Rotation Augmentation

The project also investigates the combination of flip and rotation augmentation.

The recorded image-level mean Dice scores are:

| Model | Image-Level Mean Dice |
|---|---:|
| Baseline U-Net | 0.7016 |
| Flip-only U-Net | 0.7259 |
| Flip and rotation U-Net | 0.7281 |

The flip-and-rotation model achieved the highest recorded image-level mean Dice score.

However, the patient-level statistical analysis did not establish statistically significant improvements for the comparisons involving the flip-and-rotation configuration.

In particular, the reported comparisons did not establish statistically significant differences between:

- Flip and rotation versus baseline.
- Flip and rotation versus flip-only.

This distinction is important.

Although the flip-and-rotation model achieved a slightly higher image-level mean Dice score, that numerical difference does not automatically demonstrate a statistically reliable improvement across patients.

## 12. Image-Level and Patient-Level Results Together

The results demonstrate why both image-level and patient-level evaluation are necessary.

At the image level, the flip-and-rotation configuration achieved the highest recorded mean Dice score.

At the patient level, the flip-only configuration showed a statistically significant improvement over the baseline.

The flip-and-rotation comparisons did not establish statistically significant differences.

These findings suggest that the model with the highest image-level average is not necessarily the model with the strongest statistical evidence of improvement across patients.

The distinction between numerical performance and statistical evidence is an important part of interpreting the research results.

## 13. Patient-Level Visualizations

Visualizations are used to examine how segmentation performance differs across patients.

Useful representations include:

### 13.1 Boxplots

Boxplots summarize the distribution of patient-level Dice scores.

They can show:

- Median performance.
- Interquartile range.
- Variation between patients.
- Potential outliers.

### 13.2 Paired Comparison Plots

Paired plots connect the scores of the same patients across two model configurations.

Each line represents an individual patient's change in performance.

These plots help identify whether improvements are consistent across patients or concentrated in a smaller group.

### 13.3 Patient-Level Difference Distributions

A distribution of paired differences shows how much each patient's score changed between two models.

Positive differences indicate improvement for the selected model comparison.

Negative differences indicate a reduction in performance.

**Photo location:**

`photos/statistical_analysis/`

## 14. Implementation and Saved Results

The patient-level analysis is supported by evaluation and statistical analysis scripts in:

`src/evaluation/`

The repository includes outputs related to:

- Patient-level segmentation metrics.
- Patient-level comparisons.
- Paired differences.
- Confidence intervals.
- Statistical test results.
- Model comparison visualizations.

The results are stored in:

`results/segmentation/`

Relevant outputs include patient-level metric files and statistical comparison results.

These files provide the experimental evidence for the findings described in this chapter.

## 15. Limitations of Patient-Level Analysis

Patient-level analysis provides a more appropriate perspective on observations from multiple images per individual, but it also has limitations.

### 15.1 Limited Number of Patients

The reported comparison includes 35 patients.

Although this allows paired patient-level evaluation, the number of patients remains limited.

### 15.2 Patient Variability

Patients may differ in tumor size, tumor location, image characteristics, and segmentation difficulty.

These differences can influence model performance.

### 15.3 Aggregation Method

Patient-level results depend on how image-level measurements are aggregated.

Different aggregation methods may produce different patient-level summaries.

Therefore, the aggregation method must be clearly defined and consistently applied.

### 15.4 Dataset Generalization

The patient-level findings apply to the evaluated dataset and experimental conditions.

They do not guarantee equivalent performance on patients from other datasets or clinical environments.

## 16. Chapter Summary

This chapter described the patient-level analysis performed in BrainTumorResearch.

The project investigates segmentation performance across individual patients because multiple MRI slices can originate from the same individual.

The recorded paired comparison includes 35 patients.

The flip-only U-Net achieved a mean patient-level Dice improvement of +0.0403 over the baseline, with a confidence interval of [0.0103, 0.0761] and a Holm-adjusted p-value of 0.0019.

The comparisons involving flip and rotation did not establish statistically significant improvements, despite that configuration achieving the highest recorded image-level mean Dice score.

These findings demonstrate the importance of evaluating both image-level and patient-level performance when comparing segmentation models.

The following chapter focuses on the statistical methods used to evaluate the reliability of differences between the segmentation experiments.

---

**Document:** 08 — Patient-Level Analysis  
**Project:** BrainTumorResearch  
**Evaluation:** Patient-Level Segmentation  
**Evaluated Patients:** 35  
**Main Comparison:** Flip-only U-Net vs Baseline U-Net  
**Mean Paired Dice Difference:** +0.0403  
**Confidence Interval:** [0.0103, 0.0761]  
**Holm-Adjusted p-value:** 0.0019
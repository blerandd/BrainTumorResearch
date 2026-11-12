# 09 — Statistical Analysis

## 1. Introduction

Statistical analysis is an important part of evaluating machine learning experiments.

When two models achieve different performance scores, the numerical difference alone does not establish whether one model performs reliably better than the other.

For example, one segmentation model may achieve a slightly higher Dice score than another. However, this difference could be influenced by variation between patients.

Statistical analysis helps investigate whether the observed differences provide sufficient evidence of a consistent improvement.

In BrainTumorResearch, statistical analysis is used to compare three U-Net segmentation configurations:

1. Baseline U-Net.
2. Flip-only U-Net.
3. Flip and rotation U-Net.

The analysis focuses on patient-level comparisons involving 35 patients.

The main objectives are:

- Compare segmentation performance between models.
- Calculate paired performance differences.
- Estimate confidence intervals.
- Evaluate statistical significance.
- Correct for multiple comparisons.
- Interpret the results without relying exclusively on numerical improvements.

## 2. Why Statistical Analysis Is Necessary

Consider two segmentation models:

- Model A achieves a mean Dice score of 0.72.
- Model B achieves a mean Dice score of 0.73.

Model B has a higher numerical score.

However, the difference is only:

\[
0.73 - 0.72 = 0.01
\]

This does not automatically mean that Model B is significantly better.

The difference may be small relative to the variation in performance across patients.

Statistical analysis helps determine whether the observed difference is supported by the available experimental evidence.

This is particularly important in medical image analysis, where performance can vary considerably between patients.

## 3. Paired Statistical Comparisons

The project uses paired patient-level comparisons.

A paired comparison evaluates two models using corresponding observations from the same patients.

For example, suppose both Model A and Model B are evaluated on the same 35 patients.

For each patient, the difference between the two model scores is calculated.

\[
D_i = Score_{A,i} - Score_{B,i}
\]

Where:

- \(D_i\) is the difference for patient \(i\).
- \(Score_{A,i}\) is the score achieved by Model A.
- \(Score_{B,i}\) is the score achieved by Model B.

The mean paired difference is:

\[
\bar{D} = \frac{1}{n}\sum_{i=1}^{n}D_i
\]

Where \(n\) represents the number of patients.

In this project:

\[
n = 35
\]

A positive mean difference indicates that Model A achieved a higher average score than Model B.

A negative mean difference indicates that Model B achieved a higher average score.

The paired approach is appropriate because the models are evaluated on corresponding patient observations.

It also accounts for the fact that some patients may be more difficult to segment than others.

## 4. Mean Paired Difference

The mean paired difference summarizes the average change in performance between two models.

For example, the project compares flip-only U-Net against baseline U-Net.

The recorded mean difference is:

\[
\bar{D} = +0.0403
\]

The comparison is calculated as:

\[
Dice_{flip-only} - Dice_{baseline}
\]

The positive result indicates that the flip-only configuration achieved a higher average patient-level Dice score than the baseline configuration.

The mean difference is useful because it communicates the direction and magnitude of the observed improvement.

However, it should be interpreted together with the confidence interval and adjusted p-value.

## 5. Confidence Intervals

A confidence interval provides a range of plausible values for an estimated quantity.

In this project, confidence intervals are used to describe uncertainty around the estimated patient-level performance difference.

The recorded confidence interval for the flip-only versus baseline comparison is:

\[
[0.0103,\ 0.0761]
\]

The interval is entirely above zero.

This supports a positive difference in patient-level Dice performance for flip-only U-Net compared with the baseline under the reported analysis.

### 5.1 Understanding the Confidence Interval

The lower bound is:

\[
0.0103
\]

The upper bound is:

\[
0.0761
\]

The estimated mean difference is:

\[
0.0403
\]

Because zero is not included in the reported interval, the interval supports a positive difference.

A confidence interval also communicates uncertainty.

A narrower interval indicates a more precise estimate, while a wider interval indicates greater uncertainty.

### 5.2 Why Confidence Intervals Matter

A single mean difference does not show how uncertain an estimate may be.

For example, two experiments could have the same mean improvement but very different confidence intervals.

One could provide a relatively precise estimate, while the other could show substantial uncertainty.

Therefore, confidence intervals provide additional information beyond the mean score.

## 6. P-values and Statistical Significance

A p-value is used in statistical hypothesis testing.

It measures how compatible the observed data are with a specified null hypothesis, under the assumptions of the statistical test.

For a comparison between two models, the null hypothesis commonly represents no systematic difference in performance.

The alternative hypothesis represents a difference between the models.

A small p-value indicates that the observed data would be relatively unusual under the null hypothesis.

The p-value is not the probability that the null hypothesis is true.

It also does not measure the practical importance of an improvement.

### 6.1 Statistical Significance

A commonly used significance threshold is:

\[
\alpha = 0.05
\]

When the adjusted p-value is below the selected significance threshold, the comparison is considered statistically significant under that testing procedure.

When the adjusted p-value is above the threshold, the analysis does not establish a statistically significant difference.

Importantly, a non-significant result does not prove that two models are identical.

It means that the available evidence was insufficient to establish a difference under the selected analysis.

## 7. Multiple Comparisons

The project compares three segmentation configurations.

This creates three possible pairwise comparisons:

1. Flip-only versus baseline.
2. Flip and rotation versus baseline.
3. Flip and rotation versus flip-only.

Performing multiple statistical comparisons increases the risk of obtaining at least one false-positive result.

A false positive occurs when a statistical test indicates a significant difference even though the null hypothesis is true.

To address this issue, the project uses Holm correction.

## 8. Holm-Bonferroni Correction

The Holm-Bonferroni procedure is a method for controlling the family-wise error rate when multiple statistical hypotheses are tested.

The family-wise error rate represents the probability of making at least one Type I error across a group of statistical tests.

Holm correction adjusts the interpretation of p-values to account for multiple comparisons.

### 8.1 How Holm Correction Works

Suppose there are three p-values:

\[
p_1,\ p_2,\ p_3
\]

The procedure first sorts them from smallest to largest.

\[
p_{(1)} \leq p_{(2)} \leq p_{(3)}
\]

The ordered p-values are then compared sequentially against adjusted significance thresholds.

For three comparisons, the thresholds are:

\[
\frac{\alpha}{3},\quad
\frac{\alpha}{2},\quad
\alpha
\]

For a significance level of 0.05:

\[
\frac{0.05}{3} \approx 0.0167
\]

\[
\frac{0.05}{2} = 0.025
\]

\[
0.05 = 0.05
\]

The procedure evaluates the ordered comparisons while controlling the family-wise error rate.

Holm correction is generally less conservative than applying the ordinary Bonferroni correction to every comparison.

### 8.2 Why It Is Used in This Project

Because the three U-Net configurations generate multiple pairwise comparisons, it is important to account for multiple testing.

Holm correction helps reduce the risk of interpreting a chance result as a reliable improvement.

The reported adjusted p-values should therefore be used when interpreting the statistical comparisons.

## 9. Bootstrap Confidence Intervals

Bootstrap methods estimate uncertainty by repeatedly resampling observations from the available data.

In a patient-level analysis, resampling should respect the patient as the unit of observation.

The general procedure is:

1. Begin with the available patient-level observations.
2. Randomly sample patients with replacement.
3. Calculate the statistic of interest for the sampled observations.
4. Repeat the procedure many times.
5. Examine the resulting distribution of estimates.
6. Calculate a confidence interval from the bootstrap distribution.

Sampling with replacement means that a patient can appear more than once in a bootstrap sample, while another patient may not appear in that particular sample.

Bootstrap confidence intervals are useful because they provide an estimate of uncertainty without relying exclusively on a simple theoretical approximation.

The exact number of bootstrap repetitions and interval construction method should be taken from the project's statistical analysis scripts.

## 10. Recorded Statistical Results

The project includes a paired patient-level comparison involving 35 patients.

The main recorded statistically significant comparison is flip-only U-Net versus baseline U-Net.

| Measurement | Result |
|---|---:|
| Evaluated patients | 35 |
| Mean paired Dice difference | +0.0403 |
| Confidence interval | [0.0103, 0.0761] |
| Holm-adjusted p-value | 0.0019 |

The difference is calculated as:

\[
Dice_{flip-only} - Dice_{baseline}
\]

### 10.1 Interpretation

The mean paired difference is positive.

This indicates that flip-only U-Net achieved a higher average patient-level Dice score than baseline U-Net.

The confidence interval is entirely above zero.

The Holm-adjusted p-value is:

\[
p = 0.0019
\]

Since:

\[
0.0019 < 0.05
\]

the comparison is statistically significant at the 0.05 significance level after the reported Holm correction.

This provides evidence of improved patient-level segmentation performance for the flip-only configuration compared with the baseline.

## 11. Statistical Interpretation of Flip and Rotation

The flip-and-rotation configuration achieved the highest recorded image-level mean Dice score.

| Model | Image-Level Mean Dice |
|---|---:|
| Baseline U-Net | 0.7016 |
| Flip-only U-Net | 0.7259 |
| Flip and rotation U-Net | 0.7281 |

However, the patient-level statistical comparisons involving flip and rotation did not establish statistically significant differences.

The reported comparisons were:

- Flip and rotation versus baseline.
- Flip and rotation versus flip-only.

This means that the slightly higher image-level Dice score of the flip-and-rotation configuration cannot automatically be interpreted as a statistically established improvement across patients.

The distinction between numerical improvement and statistical significance is important.

A model can achieve a higher average score without providing sufficient statistical evidence that the improvement is consistent across patients.

## 12. Statistical Significance vs Practical Importance

Statistical significance and practical importance are related but different concepts.

### Statistical Significance

Statistical significance indicates whether the observed result provides evidence against the null hypothesis under the selected testing procedure.

### Practical Importance

Practical importance considers whether the size of the improvement is meaningful in the context of the application.

For example, a small Dice improvement may be statistically significant but have limited practical impact.

Conversely, a potentially useful improvement may fail to reach statistical significance when the number of evaluated patients is limited.

Therefore, model evaluation should consider:

- Mean performance difference.
- Confidence interval.
- Adjusted p-value.
- Variation across patients.
- Potential practical relevance.

The project considers these aspects when interpreting segmentation performance.

## 13. Statistical Analysis and Patient Variability

Medical image segmentation is affected by differences between patients.

Some patients may have:

- Larger tumor regions.
- Smaller tumor regions.
- Different tumor locations.
- Irregular tumor boundaries.
- More difficult MRI characteristics.

These differences can influence segmentation performance.

Patient-level paired comparisons help examine whether a model improves performance across individuals rather than relying only on an overall image-level average.

This is especially relevant when multiple MRI slices belong to the same patient.

## 14. Reproducibility of Statistical Results

Statistical analysis should be reproducible.

For this reason, the project stores evaluation outputs and statistical results in the repository.

Relevant result files include:

- `patient_segmentation_metrics.csv`
- Patient-level comparison outputs.
- Paired difference results.
- Bootstrap confidence interval outputs.
- Statistical comparison visualizations.

The statistical analysis scripts are located within the project's evaluation and analysis source directories.

The exact test implementation, bootstrap settings, and confidence interval construction should be verified directly from the corresponding scripts before being reported in a formal publication.

This ensures that the documentation accurately describes the implemented methodology.

## 15. Limitations of the Statistical Analysis

Several limitations should be considered when interpreting the results.

### 15.1 Limited Patient Sample

The recorded paired comparisons include 35 patients.

A larger and more diverse patient sample could provide additional evidence about model performance.

### 15.2 Dataset Dependence

The statistical findings are based on the evaluated dataset and experimental conditions.

They do not guarantee equivalent results on other MRI datasets.

### 15.3 Multiple Comparisons

Multiple model comparisons require correction to control the family-wise error rate.

The project addresses this through Holm correction.

### 15.4 Statistical Power

A non-significant result does not necessarily mean that two models have identical performance.

The number of patients and the variability of their scores influence the ability to detect differences.

### 15.5 Statistical and Clinical Relevance

A statistically significant improvement does not automatically establish clinical usefulness.

Further validation would be required before making clinical claims.

## 16. Chapter Summary

This chapter explained the statistical methods used to interpret segmentation model comparisons in BrainTumorResearch.

The project uses paired patient-level comparisons involving 35 patients.

The main statistically significant result is the comparison between flip-only U-Net and baseline U-Net.

The flip-only configuration achieved:

- Mean paired Dice improvement: +0.0403.
- Confidence interval: [0.0103, 0.0761].
- Holm-adjusted p-value: 0.0019.

The comparison therefore supports a statistically significant improvement over the baseline under the reported analysis.

The flip-and-rotation configuration achieved the highest recorded image-level mean Dice score, but its patient-level comparisons did not establish statistically significant differences.

This demonstrates why statistical analysis is necessary when comparing machine learning models.

Numerical performance differences should be interpreted together with confidence intervals, adjusted p-values, patient variability, and practical relevance.

The next chapter focuses on explainability and Grad-CAM, examining how the classification models identify image regions that contribute to their predictions.

---

**Document:** 09 — Statistical Analysis  
**Project:** BrainTumorResearch  
**Statistical Unit:** Patient  
**Number of Patients:** 35  
**Multiple-Comparison Correction:** Holm-Bonferroni  
**Main Significant Comparison:** Flip-only U-Net vs Baseline U-Net  
**Mean Paired Difference:** +0.0403  
**Confidence Interval:** [0.0103, 0.0761]  
**Holm-Adjusted p-value:** 0.0019
# 14 — Limitations and Future Work

## 1. Introduction

Every scientific research project has limitations.

Identifying these limitations is important because it helps explain the conditions under which the experimental results were obtained and prevents conclusions from extending beyond the available evidence.

BrainTumorResearch investigates deep learning techniques for brain tumor classification and segmentation using MRI images.

The project includes classification models, U-Net segmentation experiments, patient-level evaluation, statistical analysis, Grad-CAM explainability, and failure case investigation.

Although the experiments provide useful findings, several limitations should be considered when interpreting the results.

This chapter discusses the main limitations of the research and identifies possible directions for future development.

## 2. Dataset Limitations

The project uses the brain tumor MRI dataset available through Figshare.

The dataset contains:

- 3,064 MRI images.
- 233 patients.
- Three tumor categories:
  - Glioma.
  - Meningioma.
  - Pituitary tumor.

Although the dataset provides a useful foundation for classification and segmentation experiments, it has several limitations.

### 2.1 Limited Number of Patients

The dataset contains 233 patients.

Although it includes multiple MRI images, the number of individual patients is more limited than the total number of images suggests.

This is important because multiple slices from the same patient are related observations.

A larger number of independent patients could provide stronger evidence about how well the models generalize to previously unseen individuals.

### 2.2 Dataset Diversity

The experiments are based on one dataset.

MRI images from other hospitals, scanners, acquisition protocols, and patient populations may contain different characteristics.

A model that performs well on one dataset may not achieve the same performance on another.

Therefore, evaluation using independent external datasets would be valuable.

### 2.3 Class Distribution

The dataset contains different numbers of images across the three tumor categories.

The recorded distribution is:

| Tumor Category | Number of Images |
|---|---:|
| Glioma | 1,426 |
| Meningioma | 708 |
| Pituitary | 930 |
| **Total** | **3,064** |

The difference in class representation should be considered when interpreting classification performance.

Accuracy alone may not fully describe how well a model performs on each category.

For this reason, the research also considers class-specific metrics and macro-averaged evaluation.

### 2.4 Image Resolution

The project preprocesses MRI images to a resolution of:

\[
256 \times 256
\]

Resizing images makes them easier to process using a consistent model input configuration.

However, resizing may reduce some fine spatial details.

This can be particularly relevant for segmentation, where accurate tumor boundaries are important.

### 2.5 Ground-Truth Annotations

Segmentation models depend on the quality of their reference masks.

The ground-truth masks provide the reference against which predicted tumor regions are evaluated.

If annotations contain uncertainty or inconsistencies, model evaluation may also be affected.

Further investigation of annotation quality and inter-annotator agreement could provide additional insight.

## 3. Classification Model Limitations

The classification component investigates CNN architectures, including:

- Custom CNN.
- VGG16.
- ResNet50.

The recorded classification evaluation achieved an accuracy of 0.8466 and a macro ROC-AUC of 0.9598 on 489 test images.

These results provide useful information about classification performance, but several limitations remain.

### 3.1 Generalization to External Data

The classification models are evaluated using the project's designated test partition.

Performance on this test set does not guarantee equivalent performance on MRI images from other institutions or datasets.

External validation would be necessary to investigate generalization beyond the original dataset.

### 3.2 Misclassification

The models may confuse visually similar tumor categories.

Incorrect predictions can occur even when the overall classification metrics are strong.

Further analysis of incorrect predictions may help identify which tumor categories are more difficult to distinguish.

### 3.3 Dependence on Training Data

Deep learning models learn patterns from their training data.

If certain visual characteristics are underrepresented, the model may perform less consistently when encountering those patterns.

Increasing dataset diversity could help address this limitation.

### 3.4 Model Interpretability

CNN architectures can be difficult to interpret.

Grad-CAM provides visual explanations of class-specific model activations, but it does not fully explain the internal decision-making process.

Additional explainability methods could provide complementary information.

## 4. Segmentation Model Limitations

The segmentation component investigates three U-Net configurations:

1. Baseline U-Net.
2. Flip-only U-Net.
3. Flip and rotation U-Net.

The recorded image-level mean Dice scores are:

| Model | Mean Dice |
|---|---:|
| Baseline U-Net | 0.7016 |
| Flip-only U-Net | 0.7259 |
| Flip and rotation U-Net | 0.7281 |

Although the augmented configurations achieved higher image-level mean Dice scores, several limitations remain.

### 4.1 Segmentation Accuracy

A Dice score provides information about the overlap between predicted and reference tumor regions.

However, a single overlap metric cannot describe every aspect of segmentation quality.

For example, two masks may achieve similar Dice scores while having different boundary characteristics.

Using additional metrics such as IoU and boundary-related measures provides a more complete evaluation.

### 4.2 Tumor Boundary Complexity

Tumor boundaries can be irregular and difficult to distinguish from surrounding tissue.

A model may correctly identify most of the tumor while producing inaccurate boundaries in certain regions.

Further work could investigate architectures and loss functions designed to improve boundary accuracy.

### 4.3 Small Tumor Regions

Small tumor regions can be difficult to segment accurately.

Even a relatively small number of incorrectly predicted pixels can have a noticeable effect on overlap metrics when the reference region is small.

Future experiments could investigate whether specialized loss functions or training strategies improve performance on smaller regions.

### 4.4 Data Augmentation

The experiments investigate flipping and rotation augmentation.

However, these are only two possible augmentation strategies.

Other approaches could be investigated, including intensity transformations, contrast adjustments, noise augmentation, and elastic transformations.

Any additional augmentation must be medically appropriate and should be evaluated carefully.

### 4.5 Model Architecture

U-Net provides an effective encoder-decoder architecture for segmentation.

However, alternative architectures may capture different spatial and contextual information.

Future work could compare U-Net with other segmentation architectures.

## 5. Patient-Level Evaluation Limitations

The project includes paired patient-level statistical comparisons involving 35 patients.

Patient-level evaluation is important because multiple MRI images may originate from the same individual.

However, the number of patients included in the recorded paired comparison remains limited.

A larger patient-level evaluation could provide stronger evidence about consistency across individuals.

Patient-level performance can also vary because patients differ in:

- Tumor size.
- Tumor location.
- Tumor shape.
- Image characteristics.
- Segmentation difficulty.

Future work could investigate these factors more systematically.

It would also be useful to examine whether model performance differs according to specific tumor characteristics.

## 6. Statistical Analysis Limitations

The project uses paired patient-level comparisons and Holm correction to investigate differences between segmentation configurations.

The recorded comparison between flip-only U-Net and baseline U-Net showed:

- Mean paired Dice difference: +0.0403.
- Confidence interval: [0.0103, 0.0761].
- Holm-adjusted p-value: 0.0019.

This comparison provides evidence of a statistically significant improvement under the reported analysis.

However, statistical significance does not automatically imply clinical importance.

### 6.1 Limited Sample Size

The paired comparison includes 35 patients.

A larger sample could provide additional information about the stability of the estimated differences.

### 6.2 Multiple Comparisons

The project investigates multiple model comparisons.

Holm correction helps control the family-wise error rate, but statistical conclusions remain dependent on the selected comparisons and testing procedure.

### 6.3 Statistical and Practical Importance

A statistically significant difference does not automatically establish that the improvement is practically important.

Future work could investigate the relationship between segmentation improvements and their potential practical significance.

### 6.4 External Statistical Validation

Future studies could repeat the comparisons using independent patient cohorts.

This would help determine whether the observed findings generalize beyond the current experimental dataset.

## 7. Explainability Limitations

The project uses Grad-CAM to investigate classification model predictions.

Although Grad-CAM provides useful visual explanations, it has several limitations.

### 7.1 Approximate Localization

Grad-CAM produces activation heatmaps rather than precise tumor boundaries.

The highlighted regions should not be interpreted as exact segmentation masks.

### 7.2 Dependence on Model Layers

The resulting visualization depends on the selected convolutional layer.

Different layers may produce different heatmap characteristics.

### 7.3 Interpretation of Heatmaps

A highlighted region does not prove that the model has learned a medically correct decision process.

Additional experiments are necessary to evaluate the reliability of explanations.

### 7.4 Future Explainability Methods

Future research could investigate additional explainability techniques and compare their results with Grad-CAM.

This could provide a broader understanding of how the classification models use image information.

## 8. Computational Limitations

The recorded development environment uses an Apple M2 processor with ARM64 architecture.

TensorFlow did not detect a GPU in the previously recorded environment.

This means the recorded experiments were not using a TensorFlow GPU device.

Training deep learning models can require considerable computational resources.

The training time depends on:

- Model architecture.
- Number of parameters.
- Image resolution.
- Batch size.
- Number of training epochs.
- Available hardware.

### 8.1 Training Time

CPU-based training may take longer than training on a suitable GPU.

This can limit the number of experiments that can be performed.

### 8.2 Hardware Compatibility

Different hardware architectures and software configurations may produce different execution times and numerical behavior.

Future experiments could investigate additional hardware configurations.

### 8.3 Model Efficiency

Future work could investigate more computationally efficient architectures and optimization strategies.

The objective would be to reduce training and inference costs while maintaining segmentation and classification performance.

## 9. Future Work

Several directions could extend the current research.

### 9.1 External Dataset Validation

One important direction is evaluating the trained models on independent MRI datasets.

This would provide additional evidence about generalization to different patient populations and imaging environments.

### 9.2 Increasing Patient Diversity

Future experiments could include more patients and additional MRI acquisition sources.

A more diverse dataset could help investigate the robustness of the models.

### 9.3 Additional Data Augmentation

Further augmentation strategies could be explored.

Potential experiments include:

- Intensity transformations.
- Contrast adjustments.
- Gaussian noise.
- Elastic transformations.
- Additional geometrical transformations.

Each strategy should be evaluated independently before being combined with other augmentation methods.

### 9.4 Alternative Segmentation Architectures

Future research could compare U-Net with alternative segmentation approaches.

Possible directions include:

- U-Net variants.
- Attention U-Net.
- Residual U-Net.
- Transformer-based segmentation architectures.

These comparisons could investigate whether additional architectural components improve tumor localization.

### 9.5 Improved Segmentation Loss Functions

Future work could investigate alternative loss functions.

Examples include:

- Dice loss.
- Focal loss.
- Tversky loss.
- Combined Dice and cross-entropy losses.

Different loss functions may influence how models handle class imbalance and difficult segmentation regions.

### 9.6 Improved Boundary Accuracy

Tumor boundary accuracy is an important aspect of medical image segmentation.

Future research could investigate boundary-aware loss functions and additional geometric evaluation methods.

This could help improve segmentation quality beyond overall overlap measurements.

### 9.7 More Detailed Patient-Level Analysis

Future experiments could investigate relationships between patient characteristics and segmentation performance.

For example, analysis could examine whether model performance changes according to tumor size, tumor category, or segmentation difficulty.

A larger patient sample would be valuable for this work.

### 9.8 Additional Explainability Methods

Future research could compare Grad-CAM with other explainability approaches.

The objective would be to investigate whether different methods provide consistent information about the regions influencing model predictions.

### 9.9 Model Ensemble Methods

Another possible direction is combining predictions from multiple models.

For classification, ensemble methods could combine the predictions of different CNN architectures.

For segmentation, predictions from different U-Net configurations could be investigated.

However, ensemble performance would need to be evaluated independently rather than assumed to be better.

### 9.10 Model Optimization

Future work could investigate model optimization techniques such as:

- Model pruning.
- Quantization.
- Knowledge distillation.
- Efficient model architectures.

These approaches could help reduce computational requirements.

## 10. Research Improvements

The following table summarizes possible improvements and their objectives.

| Research Area | Possible Improvement | Main Objective |
|---|---|---|
| Dataset | External MRI datasets | Improve generalization evaluation |
| Dataset | More independent patients | Increase patient diversity |
| Classification | Alternative CNN architectures | Investigate classification performance |
| Segmentation | Attention U-Net | Explore improved feature representation |
| Segmentation | Alternative loss functions | Improve tumor region overlap |
| Segmentation | Boundary-aware methods | Improve boundary accuracy |
| Evaluation | Larger patient-level comparisons | Strengthen statistical evidence |
| Explainability | Additional XAI methods | Improve interpretation |
| Efficiency | Model optimization | Reduce computational requirements |

## 11. Overall Research Perspective

BrainTumorResearch provides an experimental investigation of deep learning techniques for brain tumor classification and segmentation.

The results demonstrate that CNN-based models can learn patterns useful for distinguishing between three tumor categories.

The U-Net experiments also demonstrate that augmentation strategies can influence segmentation performance.

The patient-level statistical analysis provides additional evidence about differences between the segmentation configurations.

However, the research remains dependent on its dataset, model configurations, and experimental conditions.

The limitations identified in this chapter highlight areas where additional investigation could strengthen the findings.

Future work should prioritize external validation, larger patient cohorts, improved segmentation methods, and more extensive evaluation.

The objective is not simply to obtain higher numerical scores, but to develop models whose performance is more reliable, interpretable, and reproducible.

## 12. Final Chapter Summary

This chapter discussed the limitations and possible future directions of BrainTumorResearch.

The main limitations relate to:

- Dataset size and diversity.
- Classification generalization.
- Segmentation accuracy.
- Patient-level sample size.
- Statistical uncertainty.
- Explainability.
- Computational resources.
- External clinical validation.

Possible future directions include external dataset validation, additional augmentation strategies, alternative segmentation architectures, improved loss functions, more detailed patient-level analysis, and additional explainability methods.

Recognizing these limitations is an important part of scientific research.

It ensures that the experimental findings are interpreted responsibly and provides a foundation for future improvements.

---

**Document:** 14 — Limitations and Future Work  
**Project:** BrainTumorResearch  
**Main Research Areas:** Classification and Segmentation  
**Main Limitations:** Dataset diversity, patient sample size, model generalization, computational resources  
**Future Directions:** External validation, improved architectures, additional augmentation, explainability, and model optimization
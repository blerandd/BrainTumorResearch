# 10 — Explainability and Grad-CAM

## 1. Introduction

Deep learning models can achieve strong classification performance while remaining difficult to interpret.

A convolutional neural network (CNN), for example, can learn patterns from MRI images and predict whether a tumor belongs to a particular class.

However, the prediction alone does not explain which regions of the image influenced the model's decision.

This creates an important challenge in medical image analysis.

In medical applications, understanding how a model reaches a prediction is valuable because researchers need to investigate whether the model focuses on meaningful image features rather than irrelevant patterns.

BrainTumorResearch includes explainability analysis using Grad-CAM (Gradient-weighted Class Activation Mapping).

Grad-CAM produces visual heatmaps that help identify image regions contributing to a model's classification prediction.

The main objective of this chapter is to explain the theory behind Grad-CAM, its implementation concept, and its role in interpreting brain tumor classification results.

## 2. What Is Explainable Artificial Intelligence?

Explainable Artificial Intelligence (XAI) refers to methods that help humans understand how artificial intelligence systems produce their predictions.

Traditional deep learning models are often described as black-box systems.

This means that although a model receives input data and produces an output, its internal decision-making process may not be directly understandable.

For example, a brain tumor classification model may predict:

- Glioma.
- Meningioma.
- Pituitary tumor.

However, the predicted class does not tell us which parts of the MRI image influenced the decision.

Explainability methods attempt to provide additional information about the model's behavior.

In medical image analysis, this can help researchers:

- Visualize important image regions.
- Investigate model predictions.
- Identify potentially irrelevant attention patterns.
- Examine incorrect classifications.
- Improve understanding of model behavior.

Explainability does not prove that a model is correct or clinically reliable. It provides an additional way to investigate its predictions.

## 3. What Is Grad-CAM?

Grad-CAM stands for Gradient-weighted Class Activation Mapping.

It is a visualization technique designed to identify image regions that contribute to a particular class prediction made by a CNN.

Grad-CAM uses gradients flowing from a target class score into the feature maps of a convolutional layer.

These gradients indicate how strongly different feature maps influence the selected class score.

The method uses this information to produce a class-specific localization map.

The resulting heatmap can be displayed over the original MRI image.

Regions with stronger activation are highlighted more prominently.

This helps researchers investigate which areas of an image contribute to a model's prediction.

## 4. Why Grad-CAM Is Useful for Brain Tumor Classification

Brain MRI images contain complex anatomical structures and different tissue characteristics.

A classification model learns patterns from these images during training.

However, a high classification score does not automatically demonstrate that the model has learned medically meaningful features.

For example, a model might rely on:

- Tumor-related visual patterns.
- Image intensity characteristics.
- Anatomical structures.
- Background artifacts.
- Irrelevant image regions.

Grad-CAM helps visualize the spatial regions that influence the prediction.

For BrainTumorResearch, this provides an additional perspective on how the classification model distinguishes between glioma, meningioma, and pituitary tumors.

It also supports qualitative inspection of model behavior.

## 5. How Grad-CAM Works

Grad-CAM uses information from the convolutional feature maps of a trained neural network.

The process can be understood through several steps.

### Step 1 — Input MRI Image

The MRI image is passed through the trained classification model.

The model processes the image through convolutional layers and generates class scores.

### Step 2 — Select a Target Class

A target class is selected for the explanation.

For example, if the model predicts glioma, Grad-CAM can generate a heatmap explaining the model's glioma score.

The selected class determines which prediction is being investigated.

### Step 3 — Extract Feature Maps

A convolutional layer is selected, usually a layer near the end of the convolutional feature extraction network.

This layer contains spatial feature maps that represent information extracted from the input image.

These feature maps preserve spatial information about the image.

### Step 4 — Calculate Gradients

Grad-CAM calculates the gradients of the target class score with respect to the selected convolutional feature maps.

The gradients indicate how changes in different feature maps affect the target class score.

### Step 5 — Calculate Feature Map Weights

The gradients are globally averaged over their spatial dimensions.

These averages become weights representing the importance of each feature map for the selected class.

For feature map \(A^k\), the weight is:

\[
\alpha_k^c =
\frac{1}{Z}
\sum_i \sum_j
\frac{\partial y^c}{\partial A_{ij}^k}
\]

Where:

- \(\alpha_k^c\) is the importance weight for feature map \(k\) and class \(c\).
- \(y^c\) is the score for the target class.
- \(A_{ij}^k\) represents the activation at spatial position \(i,j\) in feature map \(k\).
- \(Z\) represents the number of spatial positions.

### Step 6 — Generate the Grad-CAM Heatmap

The weighted feature maps are combined.

A ReLU activation is applied to retain positive contributions:

\[
L_{Grad-CAM}^c =
ReLU\left(
\sum_k \alpha_k^c A^k
\right)
\]

The resulting heatmap highlights regions that positively contribute to the selected class score.

### Step 7 — Overlay the Heatmap

The heatmap is resized to match the original MRI image dimensions.

It can then be overlaid on the original image using a color representation.

The combined visualization allows researchers to inspect the regions emphasized by the model.

## 6. Understanding the Grad-CAM Heatmap

A Grad-CAM visualization typically contains two main elements:

1. The original MRI image.
2. A heatmap representing the model's class-specific activation.

The heatmap uses color intensity to represent the relative strength of activation.

Depending on the selected color map:

- Warmer colors generally indicate stronger activation.
- Cooler colors generally indicate weaker activation.

The exact colors depend on the visualization settings.

A region with strong activation indicates that it contributes positively to the selected class score according to the Grad-CAM calculation.

However, a highlighted region should not automatically be interpreted as the exact tumor boundary.

Grad-CAM is an explanation of model sensitivity to image features, not a tumor segmentation method.

## 7. Grad-CAM in BrainTumorResearch

The BrainTumorResearch project includes explainability outputs under:

`results/explainability/`

These outputs support the visual investigation of classification model predictions.

The purpose is to examine whether the classification model focuses on relevant regions of brain MRI images when predicting tumor categories.

The project uses explainability visualizations as an additional evaluation component alongside quantitative classification metrics.

The main classification categories are:

- Glioma.
- Meningioma.
- Pituitary tumor.

Grad-CAM helps researchers inspect the model's spatial attention patterns for individual MRI images.

The saved visualizations should be interpreted together with the original MRI images and their corresponding labels.

## 8. Interpreting Classification Explanations

When examining a Grad-CAM visualization, several questions are useful.

### 8.1 Does the Model Focus on the Tumor Region?

One important question is whether the heatmap highlights regions associated with the visible tumor.

If the model emphasizes a relevant tumor region, this may provide evidence that it is using meaningful image information.

However, visual overlap alone does not prove that the model has learned a medically correct decision process.

### 8.2 Does the Model Focus on Irrelevant Regions?

A heatmap may also highlight areas outside the visible tumor.

Possible explanations include:

- Anatomical structures.
- Image artifacts.
- Background information.
- Other image characteristics learned during training.

These regions should be investigated rather than automatically considered incorrect.

### 8.3 Does the Explanation Change Between Classes?

Grad-CAM is class-specific.

The explanation for a glioma prediction may differ from the explanation for a meningioma or pituitary prediction.

Comparing class-specific heatmaps can help researchers understand which image regions influence different class scores.

### 8.4 Does the Explanation Agree with the Prediction?

A useful analysis compares:

- The original MRI image.
- The true class.
- The predicted class.
- The prediction confidence, if available.
- The Grad-CAM heatmap.

This makes it easier to investigate both correct and incorrect classifications.

## 9. Grad-CAM and Model Evaluation

Grad-CAM complements quantitative classification metrics.

The classification evaluation measures how accurately the model predicts tumor categories.

Grad-CAM provides a visual explanation of the regions contributing to those predictions.

These approaches answer different questions.

| Evaluation Method | Main Question |
|---|---|
| Accuracy | How many predictions are correct? |
| Precision | How reliable are positive predictions for a class? |
| Recall | How many actual class examples are identified? |
| F1-score | What is the balance between precision and recall? |
| ROC-AUC | How well does the model distinguish classes across thresholds? |
| Grad-CAM | Which image regions contribute to a selected class prediction? |

Combining quantitative metrics with explainability provides a more complete evaluation of the classification system.

## 10. Grad-CAM and Incorrect Predictions

Explainability can also be useful when investigating classification errors.

Suppose an MRI image belongs to the glioma class, but the model predicts meningioma.

Grad-CAM can be generated for the predicted class and, when appropriate, the true class.

This allows researchers to examine which image regions contribute to the model's prediction.

An incorrect prediction may involve:

- Attention to irrelevant image regions.
- Overlap between visual characteristics of tumor categories.
- Unusual MRI appearance.
- Insufficiently represented patterns in the training data.

These are possible explanations to investigate, not conclusions that can be established from a heatmap alone.

Grad-CAM should therefore be used alongside the original image, ground-truth label, prediction results, and other evaluation evidence.

## 11. Limitations of Grad-CAM

Although Grad-CAM is useful, it has several limitations.

### 11.1 It Is Not a Segmentation Method

Grad-CAM does not produce an exact tumor boundary.

Its heatmap highlights regions contributing to a classification score.

A segmentation model such as U-Net is designed to predict pixel-level tumor masks.

These methods have different purposes.

### 11.2 Limited Spatial Resolution

Grad-CAM uses feature maps from convolutional layers.

These feature maps may have lower spatial resolution than the original MRI image.

As a result, heatmaps can appear relatively coarse.

### 11.3 It Does Not Prove Causality

A highlighted region indicates a positive contribution according to the Grad-CAM calculation.

It does not prove that the highlighted region is the only reason for the prediction.

### 11.4 Dependence on the Selected Layer

The selected convolutional layer influences the resulting heatmap.

Different layers can produce explanations with different levels of spatial detail and semantic information.

### 11.5 Visual Interpretation Is Subjective

Two researchers may interpret the same heatmap differently.

Therefore, visual explanations should be combined with quantitative evaluation and clearly defined analysis procedures.

### 11.6 Not Clinical Validation

A visually convincing heatmap does not establish that a model is safe or suitable for clinical use.

Additional validation using independent datasets and clinical expertise would be necessary.

## 12. Explainability and Medical AI

Explainability is particularly important when applying deep learning to medical image analysis.

Medical AI systems should not be evaluated only by their prediction accuracy.

Researchers should also investigate how the models behave and whether their predictions are supported by meaningful image information.

Grad-CAM contributes to this process by providing a visual representation of class-specific model activations.

It can help identify potential problems and guide further investigation.

However, explainability methods are supporting tools rather than substitutes for clinical validation.

## 13. Photo Documentation

The project stores explainability outputs under:

`results/explainability/`

Selected Grad-CAM images can be copied into:

`docs/photos/gradcam/`

Recommended documentation images include:

1. Original MRI image alongside its Grad-CAM heatmap.
2. Correctly classified MRI example with its explanation.
3. Incorrectly classified MRI example with its explanation.
4. Examples from different tumor categories.
5. A comparison of explanations for different class predictions, if available.

Use the actual saved visualizations from the project.

Do not create illustrative examples and present them as experimental results.

## 14. Chapter Summary

This chapter introduced explainability in deep learning and described the Grad-CAM technique used in BrainTumorResearch.

Grad-CAM generates class-specific heatmaps by examining gradients flowing into convolutional feature maps.

These heatmaps help visualize which image regions contribute to a classification model's prediction.

In this project, Grad-CAM supports the investigation of brain tumor classification predictions for glioma, meningioma, and pituitary tumor categories.

It complements quantitative evaluation metrics by providing a visual perspective on model behavior.

However, Grad-CAM does not produce exact tumor boundaries, prove that a model is correct, or establish clinical reliability.

It should be interpreted together with classification metrics, original MRI images, ground-truth labels, and segmentation results.

The following chapter focuses on failure case analysis, examining examples where the models produce incorrect predictions or weaker segmentation results.

---

**Document:** 10 — Explainability and Grad-CAM  
**Project:** BrainTumorResearch  
**Explainability Method:** Grad-CAM  
**Main Application:** Brain Tumor Classification  
**Classification Categories:** Glioma, Meningioma, Pituitary Tumor  
**Output:** Class-specific activation heatmaps
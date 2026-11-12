# 04 — Classification Models

## 1. Introduction

Brain tumor classification is one of the main components of the BrainTumorResearch project.

The objective of classification is to automatically identify the tumor category associated with an MRI image.

The project investigates deep learning architectures that can learn visual patterns from brain MRI images and use these patterns to distinguish between three tumor categories:

- Glioma
- Meningioma
- Pituitary tumor

Classification is performed using Convolutional Neural Networks (CNNs) and established deep learning architectures.

The research compares three approaches:

1. Custom CNN.
2. VGG16.
3. ResNet50.

The models are evaluated using classification metrics and visualizations to understand their predictive performance.

## 2. Classification Problem

The classification task is a supervised multi-class image classification problem.

Each MRI image is provided to a trained neural network, which produces a prediction corresponding to one of the three tumor categories.

The input is a grayscale MRI image with the target dimensions:

\[
256 \times 256 \times 1
\]

The output represents the predicted probabilities for the three classes.

The model selects the class with the highest predicted probability.

Mathematically, the predicted class can be represented as:

\[
\hat{y} = \operatorname{argmax}_{c \in C} P(c \mid x)
\]

Where:

- \(x\) represents the input MRI image.
- \(C\) represents the set of tumor categories.
- \(P(c \mid x)\) represents the predicted probability of class \(c\).
- \(\hat{y}\) represents the predicted class.

The three classes are:

\[
C = \{\text{glioma}, \text{meningioma}, \text{pituitary}\}
\]

## 3. Convolutional Neural Networks

Convolutional Neural Networks are deep learning architectures designed for processing grid-like data, particularly images.

CNNs are useful for image classification because they can automatically learn visual features from training examples.

Instead of manually defining all the image characteristics, the network learns patterns through training.

These patterns can include:

- Edges and boundaries.
- Textures.
- Shapes.
- Local structures.
- More complex visual representations.

### 3.1 Convolutional Layers

A convolutional layer applies learnable filters to an input image or feature map.

Each filter detects particular patterns in the input.

The resulting feature maps represent information extracted from the image.

During training, the network adjusts the filter parameters to learn features that help distinguish between the tumor categories.

### 3.2 Activation Functions

Activation functions introduce non-linearity into neural networks.

Without non-linear activation functions, a network composed only of linear operations would be limited in the complexity of patterns it could learn.

A commonly used activation function in CNN architectures is ReLU:

\[
ReLU(x) = \max(0,x)
\]

ReLU keeps positive values and replaces negative values with zero.

### 3.3 Pooling Layers

Pooling layers reduce the spatial dimensions of feature maps.

Max pooling, for example, selects the maximum value from a local region.

This reduces the amount of information that must be processed by later layers while retaining strong activation responses.

### 3.4 Classification Layers

After feature extraction, the learned features are used by classification layers to produce the final prediction.

The output layer represents the three tumor categories.

A Softmax activation can convert output scores into class probabilities:

\[
P(y_i) = \frac{e^{z_i}}{\sum_{j=1}^{3}e^{z_j}}
\]

Where:

- \(z_i\) represents the score for class \(i\).
- \(P(y_i)\) represents the predicted probability for that class.
- The denominator normalizes the scores across all three categories.

## 4. Custom CNN

The first classification approach is a custom Convolutional Neural Network developed specifically for this project.

Unlike a pretrained architecture, a custom CNN is designed and trained using the project's MRI dataset.

Its purpose is to learn useful visual representations directly from the training images.

The custom CNN provides a baseline architecture for investigating how a neural network performs on the brain tumor classification task.

### 4.1 Architecture

The custom CNN uses convolutional feature extraction followed by classification layers.

Its main architectural components include:

- Convolutional processing.
- Non-linear activation.
- Feature extraction.
- Classification layers.
- Output predictions for three tumor categories.

The exact layer configuration, number of filters, kernel sizes, and other architectural parameters should be taken from the implementation in the repository rather than assumed from a generic CNN design.

### 4.2 Training

The custom CNN is trained using the prepared MRI dataset.

During training:

1. MRI images are loaded.
2. Images are passed through the network.
3. The network produces class predictions.
4. Predictions are compared with the true labels.
5. The loss function measures prediction error.
6. Backpropagation calculates gradients.
7. The optimizer updates the network parameters.

This process is repeated across training batches and epochs.

### 4.3 Saved Model

The trained custom CNN checkpoint is stored at:

`models/classification/custom_cnn_best.keras`

The saved model can be loaded for evaluation and prediction without training the network again.

## 5. VGG16

VGG16 is a convolutional neural network architecture developed by the Visual Geometry Group at the University of Oxford.

It is known for its use of small convolutional filters, particularly 3 × 3 filters, arranged in a deep network.

VGG16 uses a sequence of convolutional layers and pooling operations to extract increasingly complex image features.

### 5.1 Architecture Characteristics

The original VGG16 architecture contains:

- Convolutional layers.
- ReLU activation functions.
- Max-pooling layers.
- Fully connected layers.
- A classification output layer.

The architecture uses a relatively simple and consistent design based on repeated convolutional operations.

### 5.2 Application to Brain MRI Classification

In this project, VGG16 is investigated as a classification architecture for brain MRI images.

The purpose is to evaluate how this established architecture performs when applied to the three-category tumor classification problem.

Because the original VGG16 architecture was designed for general image classification, its input and output configuration must be adapted to the requirements of the MRI classification task.

The exact implementation, including whether pretrained weights are used, the input-channel adaptation, and the classification head, should be verified from the corresponding project source code.

## 6. ResNet50

ResNet50 is a deep convolutional neural network architecture based on residual learning.

It was introduced to make the training of deeper neural networks more effective.

As networks become deeper, optimization can become difficult. Residual connections provide an alternative path through which information and gradients can propagate.

### 6.1 Residual Connections

A residual block learns a residual mapping rather than requiring every block to learn a complete transformation directly.

Its basic mathematical representation is:

\[
y = F(x) + x
\]

Where:

- \(x\) represents the input.
- \(F(x)\) represents the transformation learned by the residual block.
- \(y\) represents the resulting output.

The shortcut connection adds the original input to the transformed representation.

This structure helps information flow through deeper networks.

### 6.2 Architecture Characteristics

ResNet50 is a deep CNN architecture containing residual blocks and convolutional operations.

Its architecture uses bottleneck residual blocks to process image features efficiently.

### 6.3 Application to Brain MRI Classification

ResNet50 is investigated as another classification approach in this project.

Its residual connections provide a different architectural strategy compared with the custom CNN and VGG16.

The purpose of including ResNet50 is to compare the classification performance of architectures with different feature extraction and network structures.

The exact input adaptation, pretrained-weight configuration, and final classification layers should be verified from the implementation.

## 7. Classification Model Comparison

The project investigates the following architectures:

| Model | Architecture Type | Main Characteristic |
|---|---|---|
| Custom CNN | Custom convolutional network | Developed for the project |
| VGG16 | Deep CNN | Repeated small convolutional filters |
| ResNet50 | Residual CNN | Residual connections |

The comparison investigates how different architectural approaches perform on the same classification problem.

The main evaluation considerations include:

- Overall classification accuracy.
- Precision.
- Recall.
- F1-score.
- ROC-AUC.
- Confusion matrix.
- Training and validation behavior.

Using multiple evaluation metrics provides a more complete understanding of classification performance.

## 8. Classification Training Process

The classification models are trained using the prepared MRI dataset.

The dataset preparation and loading procedures are described in:

`02_dataset_and_preprocessing.md`

The training process follows these general stages:

1. Load the prepared training data.
2. Load the validation data.
3. Initialize the classification architecture.
4. Configure the loss function and optimizer.
5. Train the model using batches of MRI images.
6. Monitor training and validation performance.
7. Save the selected model checkpoint.
8. Evaluate the trained model using the test dataset.

### 8.1 Loss Function

For multi-class classification, categorical cross-entropy is a commonly used loss function.

For a single example, it can be represented as:

\[
L = -\sum_{i=1}^{3} y_i \log(\hat{y}_i)
\]

Where:

- \(y_i\) represents the true class indicator.
- \(\hat{y}_i\) represents the predicted probability for class \(i\).
- \(L\) represents the classification loss.

The exact loss configuration used by each model should be taken from its training implementation.

### 8.2 Optimizer

An optimizer updates the model parameters to minimize the training loss.

The specific optimizer and its hyperparameters are defined in the classification training code.

These details should be documented from the actual configuration used for each experiment.

### 8.3 Training and Validation Monitoring

Training and validation metrics are monitored to examine model behavior during training.

Training performance indicates how well the model learns from the training data.

Validation performance provides information about how well the model performs on data not used for parameter updates.

Training curves can help identify:

- Convergence behavior.
- Underfitting.
- Potential overfitting.
- Differences between training and validation performance.

## 9. Classification Evaluation Metrics

The classification models are evaluated using multiple metrics.

### 9.1 Accuracy

Accuracy measures the proportion of correctly classified examples.

\[
Accuracy = \frac{TP + TN}{TP + TN + FP + FN}
\]

For multi-class classification, accuracy is calculated from the total number of correct predictions divided by the total number of predictions.

### 9.2 Precision

Precision measures how many predictions assigned to a class are correct.

\[
Precision = \frac{TP}{TP + FP}
\]

A high precision indicates that the model produces relatively few false positive predictions for that class.

### 9.3 Recall

Recall measures how many actual examples belonging to a class are correctly identified.

\[
Recall = \frac{TP}{TP + FN}
\]

A high recall indicates that the model identifies a large proportion of the actual examples belonging to that class.

### 9.4 F1-Score

The F1-score is the harmonic mean of precision and recall.

\[
F1 = 2 \times \frac{Precision \times Recall}{Precision + Recall}
\]

It provides a combined measure of precision and recall.

### 9.5 ROC-AUC

The Receiver Operating Characteristic curve examines the relationship between the true positive rate and false positive rate at different classification thresholds.

The Area Under the Curve (AUC) summarizes this relationship.

For a multi-class classification problem, ROC-AUC can be calculated using a one-vs-rest approach.

A higher AUC generally indicates better separation between the classes.

### 9.6 Confusion Matrix

A confusion matrix compares the actual class labels with the predicted class labels.

For the three-category classification problem, it contains three rows and three columns.

The diagonal represents correct predictions, while off-diagonal values represent classification errors.

The confusion matrix helps identify which tumor categories are most frequently confused by the model.

## 10. Classification Results

The recorded classification evaluation for the project includes the following results:

| Metric | Recorded Result |
|---|---:|
| Test accuracy | 0.8466 |
| Macro ROC-AUC | 0.9598 |
| Test images | 489 |

The test accuracy of 0.8466 corresponds to approximately 84.66% correctly classified test images.

The macro ROC-AUC of 0.9598 indicates strong class-discrimination performance when considering the macro-averaged one-vs-rest evaluation.

These results should be interpreted together rather than treating either metric as a complete description of model performance.

The reported values should be associated with the exact model and evaluation run that produced them in the results directory.

### 10.1 Classification Visualizations

The project contains classification-related outputs, including:

- Confusion matrix.
- ROC curves.
- Training and validation curves.
- Classification metrics.
- Test predictions.
- Patient-level classification results.

These visualizations support the interpretation of classification performance.

## 11. Classification Implementation

The classification implementation is located in the `src/` directory.

Relevant source files include:

- `src/models/classification.py`
- `src/models/classification_cnn.py`
- `src/training/train_classification.py`

These files contain the model definitions and training implementation used in the classification experiments.

The saved custom CNN checkpoint is located at:

`models/classification/custom_cnn_best.keras`

Classification evaluation outputs are stored in:

`results/classification/`

## 12. Explainability Connection

Classification performance alone does not explain which regions of an MRI image influence the model's predictions.

For this reason, the project also investigates Grad-CAM.

Grad-CAM produces visual heatmaps that highlight image regions contributing to a model's classification decision.

These visualizations provide an additional way to examine model behavior.

The explainability methodology is described in:

`10_explainability_gradcam.md`

## 13. Limitations of Classification

Several limitations must be considered when interpreting classification results.

First, the dataset contains a limited number of patients compared with the variety of cases encountered in real clinical environments.

Second, the number of images differs between tumor categories.

Third, MRI images from the same patient can share visual characteristics, making patient-independent evaluation important.

Fourth, high classification performance does not automatically demonstrate that a model is suitable for clinical use.

Finally, model predictions should be interpreted as experimental outputs rather than medical diagnoses.

## 14. Chapter Summary

This chapter described the classification component of BrainTumorResearch.

The project investigates three classification architectures:

- Custom CNN.
- VGG16.
- ResNet50.

The classification task predicts one of three tumor categories: glioma, meningioma, or pituitary tumor.

The chapter explained the main principles of CNNs, convolutional feature extraction, activation functions, pooling, residual connections, classification outputs, and evaluation metrics.

The recorded classification evaluation includes a test accuracy of 0.8466 and a macro ROC-AUC of 0.9598 across 489 test images.

The next chapter focuses on the segmentation component of the project and explains the U-Net architecture and the segmentation experiments.

---

**Document:** 04 — Classification Models  
**Project:** BrainTumorResearch  
**Task:** Multi-Class Brain Tumor MRI Classification  
**Classes:** Glioma, Meningioma, Pituitary Tumor  
**Investigated Architectures:** Custom CNN, VGG16, ResNet50
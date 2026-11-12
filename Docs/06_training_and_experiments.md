# 06 — Training and Experiments

## 1. Introduction

This chapter describes the training procedures and experimental configurations used in the BrainTumorResearch project.

Training is the process through which a deep learning model learns patterns from the prepared dataset by adjusting its internal parameters.

The project investigates two main deep learning tasks:

- Brain tumor MRI classification.
- Brain tumor MRI segmentation.

Although both tasks use deep learning, their training objectives are different.

Classification models learn to predict the tumor category associated with an MRI image.

Segmentation models learn to identify the tumor region at the pixel level.

The research also investigates how different training configurations and data augmentation strategies influence model performance.

## 2. Training Environment

The project is implemented using Python and TensorFlow.

The main technologies used during model development and training include:

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| TensorFlow | Deep learning framework |
| Keras | Model construction and training |
| NumPy | Numerical operations |
| Pandas | Dataset and experiment management |
| Matplotlib | Training visualization |
| Scikit-learn | Evaluation and analysis |

The project uses a Python virtual environment to manage dependencies.

The required packages are listed in the root-level:

`requirements.txt`

The training scripts are organized inside:

`src/training/`

## 3. Dataset Preparation for Training

Before training begins, the dataset is prepared using the preprocessing pipeline described in:

`02_dataset_and_preprocessing.md`

The dataset is divided into three partitions:

| Dataset | Percentage | Purpose |
|---|---:|---|
| Training | 70% | Model parameter optimization |
| Validation | 15% | Monitoring and model selection |
| Testing | 15% | Final evaluation |

The splitting strategy is patient-independent.

This means that images belonging to the same patient are kept within the same partition.

The project uses a random seed of 42 to support reproducibility of the dataset preparation process.

The model data pipeline uses:

| Parameter | Value |
|---|---:|
| Image height | 256 |
| Image width | 256 |
| Image channels | 1 |
| Batch size | 16 |
| Random seed | 42 |

The training process uses the prepared data rather than directly loading the original dataset for every experiment.

## 4. Training Fundamentals

Deep learning training involves repeatedly passing data through a neural network and adjusting its parameters to reduce prediction error.

The main stages are:

1. Load a batch of training examples.
2. Pass the examples through the neural network.
3. Generate predictions.
4. Calculate the loss.
5. Calculate gradients using backpropagation.
6. Update model parameters using an optimizer.
7. Repeat the process for subsequent batches.

### 4.1 Epoch

An epoch represents one complete pass through the training dataset.

For example, if the training dataset contains 1,000 images, an epoch means that the model has processed all 1,000 training images once, assuming every image is included.

Training usually requires multiple epochs so that the model can progressively improve its predictions.

### 4.2 Batch Size

Batch size represents the number of training examples processed together before a parameter update.

The project data loader uses a batch size of 16.

This means that the model processes up to 16 images in a training batch.

Batch processing helps manage memory consumption and allows efficient model training.

### 4.3 Learning Rate

The learning rate controls the size of the parameter updates made by the optimizer.

A learning rate that is too high may cause unstable training.

A learning rate that is too low may make training progress very slowly.

The exact learning rate used in each experiment is defined in the corresponding training script.

### 4.4 Backpropagation

Backpropagation calculates how changes in model parameters affect the loss.

The gradients are propagated backward through the network.

The optimizer uses these gradients to update the model parameters.

This process allows the network to gradually learn patterns that improve its predictions.

## 5. Classification Training

The classification component investigates three architectures:

- Custom CNN.
- VGG16.
- ResNet50.

Each architecture is used to predict one of the three tumor categories:

- Glioma.
- Meningioma.
- Pituitary tumor.

### 5.1 Classification Training Workflow

The classification training process follows these stages:

1. Load the prepared MRI images and class labels.
2. Initialize the classification model.
3. Configure the loss function and optimizer.
4. Pass training images through the network.
5. Calculate classification loss.
6. Update model parameters.
7. Monitor validation performance.
8. Save the selected model checkpoint.
9. Evaluate the trained model on the test partition.

### 5.2 Classification Loss

Classification training uses a loss function to measure the difference between predicted class probabilities and the true labels.

For multi-class classification, categorical cross-entropy is commonly used.

Its mathematical representation is:

\[
L = -\sum_{i=1}^{C} y_i \log(\hat{y}_i)
\]

Where:

- \(C\) represents the number of classes.
- \(y_i\) represents the true class indicator.
- \(\hat{y}_i\) represents the predicted probability for class \(i\).

The exact loss configuration is determined by the classification training implementation.

### 5.3 Classification Training Implementation

The main classification training script is:

`src/training/train_classification.py`

The model definitions are located in:

- `src/models/classification.py`
- `src/models/classification_cnn.py`

The trained custom CNN checkpoint is stored at:

`models/classification/custom_cnn_best.keras`

## 6. Segmentation Training

The segmentation component uses a U-Net-based architecture.

Unlike classification, the model predicts a segmentation mask instead of a single category.

The input consists of an MRI image, while the target consists of the corresponding ground-truth tumor mask.

The model learns to identify tumor pixels by minimizing the difference between its predicted mask and the expected mask.

### 6.1 Segmentation Training Workflow

The segmentation training process follows these stages:

1. Load MRI images.
2. Load their corresponding segmentation masks.
3. Prepare image and mask dimensions.
4. Apply training transformations when enabled.
5. Pass the MRI image through the U-Net model.
6. Generate a predicted segmentation mask.
7. Calculate segmentation loss.
8. Update model parameters.
9. Monitor validation performance.
10. Save the selected model checkpoint.

### 6.2 Segmentation Loss

The project includes custom segmentation-related functions:

- `dice_coefficient`
- `dice_loss`
- `combined_segmentation_loss`

The Dice coefficient measures overlap between the predicted and ground-truth regions.

\[
Dice = \frac{2|P \cap G|}{|P| + |G|}
\]

Dice loss is commonly represented as:

\[
DiceLoss = 1 - Dice
\]

The project also includes a combined segmentation loss function.

The exact implementation and weighting of the loss components are defined in:

`src/models/unet.py`

## 7. Data Augmentation

Data augmentation is used to introduce variations into the training data.

Instead of relying exclusively on the original training images, augmentation creates transformed versions of existing examples.

The purpose is to investigate whether additional image variations can improve model generalization.

The segmentation experiments investigate flip and rotation augmentation.

### 7.1 Flip Augmentation

Flip augmentation changes the orientation of an image.

It can introduce additional spatial variations into the training data.

For segmentation, the transformation must also be applied to the corresponding ground-truth mask.

This ensures that the tumor region remains correctly aligned with the transformed MRI image.

### 7.2 Rotation Augmentation

Rotation augmentation creates additional training examples by rotating the image.

The corresponding segmentation mask must receive the same rotation.

This allows the model to encounter different spatial orientations during training.

### 7.3 Combined Augmentation

The combined experiment uses both flip and rotation augmentation.

The purpose is to investigate whether combining the transformations provides additional benefits compared with using flip augmentation alone.

### 7.4 Augmentation Limitations

Augmentation must preserve the meaningful relationship between an MRI image and its segmentation mask.

Incorrect transformations can introduce unrealistic examples or misaligned targets.

For this reason, image and mask transformations must be applied consistently.

## 8. Segmentation Experiments

The project investigates three main U-Net training configurations.

### Experiment 1 — Baseline U-Net

The baseline U-Net is trained without the additional flip and rotation augmentation strategies investigated in the other experiments.

It provides the reference model for the segmentation comparisons.

**Training script:**

`src/training/train_unet.py`

**Saved checkpoint:**

`models/segmentation/unet_best.keras`

### Experiment 2 — Flip-Only U-Net

The second experiment investigates the effect of flip augmentation.

The model is trained using transformed MRI images and their correspondingly transformed segmentation masks.

The objective is to determine whether flip augmentation improves tumor-region prediction.

**Training script:**

`src/training/train_unet_flip_only.py`

**Saved checkpoint:**

`models/segmentation/unet_flip_only_best.keras`

### Experiment 3 — Flip and Rotation U-Net

The third experiment investigates the combination of flip and rotation augmentation.

The objective is to examine whether combining these transformations provides further improvements in segmentation performance.

**Training script:**

`src/training/train_unet_augmented.py`

**Saved checkpoint:**

`models/segmentation/unet_augmented_best.keras`

## 9. Experimental Comparison

The three segmentation experiments are compared using their recorded image-level mean Dice scores.

| Experiment | Mean Dice |
|---|---:|
| Baseline U-Net | 0.7016 |
| Flip-only U-Net | 0.7259 |
| Flip and rotation U-Net | 0.7281 |

The recorded results show that both augmentation configurations achieved higher image-level mean Dice scores than the baseline.

The flip-only configuration improved the mean Dice score by:

\[
0.7259 - 0.7016 = 0.0243
\]

The flip-and-rotation configuration improved the mean Dice score by:

\[
0.7281 - 0.7016 = 0.0265
\]

The difference between flip-and-rotation and flip-only was:

\[
0.7281 - 0.7259 = 0.0022
\]

These differences describe the recorded image-level results.

They do not, by themselves, establish statistical significance.

Patient-level comparisons and statistical analysis are discussed in:

`08_patient_level_analysis.md`

`09_statistical_analysis.md`

## 10. Monitoring Training Performance

Training and validation performance are monitored throughout the training process.

Training curves provide information about how the model learns over time.

Common visualizations include:

- Training loss.
- Validation loss.
- Training metrics.
- Validation metrics.

### 10.1 Training Loss

Training loss measures prediction error on the training dataset.

A decreasing training loss generally indicates that the model is learning patterns from the training examples.

### 10.2 Validation Loss

Validation loss measures prediction error on the validation dataset.

It provides information about model performance on examples that are not used for parameter updates.

### 10.3 Overfitting

Overfitting occurs when a model learns patterns specific to the training data but performs less effectively on unseen data.

One possible indication is that training loss continues to decrease while validation loss stops improving or begins increasing.

### 10.4 Underfitting

Underfitting occurs when a model has not learned sufficiently useful patterns from the training data.

It may be reflected in poor performance on both training and validation data.

## 11. Model Checkpoints

A model checkpoint stores the model's learned parameters.

Checkpoints allow trained models to be loaded later without repeating the entire training process.

The project stores its classification and segmentation checkpoints in the `models/` directory.

The main segmentation checkpoints are:

- `unet_best.keras`
- `unet_flip_only_best.keras`
- `unet_augmented_best.keras`

The classification checkpoint includes:

- `custom_cnn_best.keras`

The `.keras` format is used for saving TensorFlow/Keras models.

## 12. Training Results and Visualizations

Training results are stored in the repository's `results/` directory.

The project includes outputs such as:

- Training history.
- Training and validation curves.
- Evaluation metrics.
- Model comparison results.
- Segmentation visualizations.
- Classification predictions.

These outputs provide evidence of the experiments and support the analysis of model performance.

**Photo locations:**

- `docs/photos/training/`
- `docs/photos/model_comparison/`

## 13. Reproducibility

Reproducibility is important because experimental results should be traceable to the data, model configuration, and training procedure that produced them.

The project supports reproducibility through:

- Organized training scripts.
- A documented preprocessing pipeline.
- Patient-independent dataset splitting.
- A fixed random seed for dataset preparation.
- Saved model checkpoints.
- Recorded training histories.
- Stored evaluation outputs.

However, exact reproduction also depends on the software environment, library versions, hardware, and training configurations.

The environment and reproduction procedures are discussed further in:

`13_reproducibility_and_environment.md`

## 14. Limitations of the Experiments

Several limitations should be considered when interpreting the experimental results.

### 14.1 Dataset Size

The number of available patients is limited compared with the diversity of medical imaging cases encountered in practice.

### 14.2 Augmentation Choices

The experiments investigate specific augmentation strategies rather than every possible image transformation.

Different transformations may produce different results.

### 14.3 Training Variability

Deep learning training can be influenced by initialization, random operations, software versions, and hardware behavior.

### 14.4 Image-Level Evaluation

Image-level metrics do not fully describe performance across individual patients.

For this reason, the project also performs patient-level analysis.

### 14.5 Statistical Interpretation

Small numerical differences between experiments should not automatically be interpreted as meaningful improvements.

Statistical comparisons are needed to examine the reliability of observed differences.

## 15. Chapter Summary

This chapter described the training procedures and experimental configurations used in BrainTumorResearch.

The project investigates both classification and segmentation training.

The classification component evaluates Custom CNN, VGG16, and ResNet50 architectures.

The segmentation component investigates three U-Net configurations:

- Baseline U-Net.
- Flip-only U-Net.
- Flip and rotation U-Net.

The recorded image-level mean Dice scores are 0.7016, 0.7259, and 0.7281, respectively.

The chapter also explained epochs, batch size, learning rate, backpropagation, loss functions, data augmentation, training monitoring, model checkpoints, and reproducibility.

The following chapter focuses on the evaluation metrics used to measure classification and segmentation performance.

---

**Document:** 06 — Training and Experiments  
**Project:** BrainTumorResearch  
**Framework:** TensorFlow / Keras  
**Classification Experiments:** Custom CNN, VGG16, ResNet50  
**Segmentation Experiments:** Baseline U-Net, Flip-only U-Net, Flip and Rotation U-Net  
**Batch Size:** 16  
**Image Resolution:** 256 × 256
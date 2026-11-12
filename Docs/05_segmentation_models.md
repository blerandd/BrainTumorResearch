# 05 — Segmentation Models

## 1. Introduction

Brain tumor segmentation is one of the main components of the BrainTumorResearch project.

While classification identifies the tumor category associated with an MRI image, segmentation focuses on identifying the exact spatial region occupied by the tumor.

Segmentation is a pixel-level image analysis task. Instead of producing a single class label, the model generates a mask that indicates which pixels belong to the tumor region.

The project investigates a U-Net-based deep learning architecture for brain tumor segmentation.

The main objective is to train a model that can learn the relationship between MRI images and their corresponding ground-truth segmentation masks.

The research also compares different training configurations to investigate how data augmentation influences segmentation performance.

## 2. Segmentation Problem

The segmentation task is formulated as a supervised image-to-image learning problem.

The input is an MRI image, and the expected output is a segmentation mask.

For each MRI image, the model predicts a corresponding mask identifying the tumor region.

The input image has the following dimensions:

\[
256 \times 256 \times 1
\]

The predicted segmentation mask represents the spatial distribution of tumor pixels.

For binary segmentation, the output distinguishes between two regions:

- Background.
- Tumor.

A predicted probability map can be converted into a binary mask using a threshold.

For example:

\[
\hat{M}(i,j) =
\begin{cases}
1 & \text{if } P(i,j) \geq t \\
0 & \text{if } P(i,j) < t
\end{cases}
\]

Where:

- \(P(i,j)\) represents the predicted tumor probability at pixel location \((i,j)\).
- \(t\) represents the selected threshold.
- \(\hat{M}(i,j)\) represents the resulting binary segmentation prediction.

The threshold determines which pixels are classified as tumor pixels.

## 3. Introduction to U-Net

U-Net is a convolutional neural network architecture designed for biomedical image segmentation.

It was introduced by Olaf Ronneberger, Philipp Fischer, and Thomas Brox in 2015.

The architecture is called U-Net because its structure resembles the letter U.

It consists of two main paths:

1. Contracting path (encoder).
2. Expanding path (decoder).

The encoder extracts increasingly complex image features while reducing spatial dimensions.

The decoder reconstructs spatial information and generates a segmentation output.

U-Net is particularly useful for biomedical segmentation because it combines contextual information with spatial details through skip connections.

## 4. U-Net Architecture

The U-Net architecture consists of several interconnected components.

### 4.1 Contracting Path — Encoder

The contracting path extracts features from the input MRI image.

It uses convolutional operations followed by non-linear activation functions and spatial downsampling.

During this process:

- Feature maps become deeper.
- Spatial dimensions become smaller.
- The network learns increasingly complex representations.

The encoder helps the network understand the overall context of the MRI image.

Early layers learn relatively simple visual patterns, while deeper layers learn more complex structures.

### 4.2 Convolutional Layers

Convolutional layers apply learnable filters to the input feature maps.

These filters detect visual patterns such as:

- Edges.
- Boundaries.
- Textures.
- Shapes.
- Anatomical structures.

The learned features are passed to subsequent layers.

### 4.3 Activation Functions

Activation functions introduce non-linearity into the network.

ReLU is commonly represented as:

\[
ReLU(x) = \max(0,x)
\]

It allows the network to learn more complex relationships between input images and segmentation masks.

### 4.4 Downsampling

Downsampling reduces the spatial dimensions of feature maps.

In U-Net, pooling operations are commonly used for this purpose.

Reducing spatial dimensions allows the network to capture larger contextual regions while increasing the depth of feature representations.

### 4.5 Expanding Path — Decoder

The expanding path reconstructs spatial information from the features extracted by the encoder.

It progressively increases the spatial dimensions of feature maps.

The decoder combines higher-level contextual information with spatial information transferred from the encoder.

This allows the network to produce detailed segmentation predictions.

### 4.6 Upsampling

Upsampling increases the spatial dimensions of feature maps.

It helps reconstruct the spatial resolution required for pixel-level predictions.

The decoder uses upsampling operations to progressively recover image dimensions.

### 4.7 Skip Connections

Skip connections connect corresponding encoder and decoder stages.

They transfer spatial information from the contracting path to the expanding path.

These connections are important because downsampling can remove fine spatial details.

By combining encoder features with decoder features, U-Net can use both:

- High-level contextual information.
- Fine-grained spatial information.

This helps the network identify tumor boundaries and smaller structures.

### 4.8 Output Layer

The final layer generates the segmentation prediction.

For binary segmentation, the output represents the probability that each pixel belongs to the tumor region.

A sigmoid activation can be used to produce values between 0 and 1.

The output is then compared with the ground-truth mask during training.

## 5. Encoder and Decoder Structure

The U-Net architecture can be summarized as follows:

| Component | Main Function |
|---|---|
| Input layer | Receives the MRI image. |
| Encoder | Extracts image features and reduces spatial dimensions. |
| Convolutional blocks | Learn visual representations. |
| Downsampling | Reduces feature-map resolution. |
| Bottleneck | Captures deeper contextual information. |
| Decoder | Reconstructs spatial information. |
| Upsampling | Increases feature-map resolution. |
| Skip connections | Transfer spatial information from encoder to decoder. |
| Output layer | Produces the predicted segmentation mask. |

The combination of the encoder, bottleneck, decoder, and skip connections enables U-Net to perform pixel-level segmentation.

## 6. Ground-Truth Segmentation Masks

The segmentation models are trained using MRI images and their corresponding ground-truth masks.

The ground-truth mask represents the expected tumor region.

During training, the model generates a predicted mask and compares it with the ground-truth mask.

The difference between the predicted and expected masks is measured using a segmentation loss function.

The model parameters are then updated to reduce this loss.

The MRI image and its mask must undergo spatially consistent preprocessing.

For example, if an MRI image is flipped or rotated during augmentation, the corresponding mask must receive the same transformation.

Otherwise, the input image and target mask would no longer represent the same spatial information.

## 7. Segmentation Loss Functions

Loss functions measure the difference between the predicted segmentation and the ground-truth mask.

The project includes custom segmentation-related functions:

- `dice_coefficient`
- `dice_loss`
- `combined_segmentation_loss`

### 7.1 Dice Coefficient

The Dice coefficient measures the overlap between the predicted mask and the ground-truth mask.

It is defined as:

\[
Dice = \frac{2|P \cap G|}{|P| + |G|}
\]

Where:

- \(P\) represents the predicted tumor region.
- \(G\) represents the ground-truth tumor region.
- \(|P \cap G|\) represents the number of overlapping pixels.

A Dice score closer to 1 indicates greater overlap.

A score closer to 0 indicates less overlap.

### 7.2 Dice Loss

Dice loss converts the overlap measure into a quantity that can be minimized during training.

A common formulation is:

\[
DiceLoss = 1 - Dice
\]

Minimizing Dice loss encourages the predicted mask to overlap more closely with the ground-truth mask.

### 7.3 Combined Segmentation Loss

The project also includes a combined segmentation loss function.

Combined losses can incorporate more than one objective to guide segmentation training.

The exact mathematical formulation and weighting used in this project should be taken from the implementation in `src/models/unet.py`.

## 8. Segmentation Experiments

The project investigates three U-Net training configurations.

### 8.1 Baseline U-Net

The baseline U-Net is the reference segmentation model.

It is trained using the prepared MRI images and corresponding segmentation masks without the additional flip and rotation augmentation strategies investigated in the other experiments.

Its performance provides a reference point for evaluating the effects of augmentation.

### 8.2 U-Net with Flip Augmentation

The second experiment investigates the use of flip augmentation.

Flipping creates transformed training examples by changing the orientation of the MRI image.

The corresponding segmentation mask must be transformed in exactly the same way.

The purpose is to expose the model to additional spatial variations during training.

This experiment investigates whether flip augmentation improves the model's ability to generalize to unseen patients.

### 8.3 U-Net with Flip and Rotation Augmentation

The third experiment combines flip augmentation with rotation augmentation.

Rotation creates additional spatial variations in the training images.

As with flipping, the same transformation must be applied to the corresponding segmentation mask.

This experiment investigates whether combining different augmentation strategies produces additional improvements in segmentation performance.

## 9. Model Comparison

The three segmentation configurations are compared using the same general segmentation task.

| Model | Training Configuration |
|---|---|
| Baseline U-Net | No additional flip or rotation augmentation |
| Flip-only U-Net | Flip augmentation |
| Flip and rotation U-Net | Flip and rotation augmentation |

The main objective is to investigate how augmentation influences tumor-region prediction.

The comparison considers:

- Image-level Dice score.
- Other segmentation metrics.
- Boundary-related metrics.
- Patient-level performance.
- Statistical differences between experiments.
- Visual quality of predicted masks.

The experiments are evaluated using the same general research framework to support meaningful comparisons.

## 10. Recorded Segmentation Results

The project contains the following recorded image-level mean Dice scores:

| Model | Mean Dice |
|---|---:|
| Baseline U-Net | 0.7016 |
| Flip-only U-Net | 0.7259 |
| Flip and rotation U-Net | 0.7281 |

These values indicate that both augmented configurations achieved higher recorded image-level mean Dice scores than the baseline.

The flip-and-rotation configuration achieved the highest image-level mean Dice score among the three configurations.

However, the difference between image-level scores does not automatically establish that one model is statistically superior.

Patient-level analysis and statistical comparisons are therefore also performed.

## 11. Saved Segmentation Models

The repository contains saved segmentation model checkpoints.

The checkpoint files are:

- `models/segmentation/unet_best.keras`
- `models/segmentation/unet_flip_only_best.keras`
- `models/segmentation/unet_augmented_best.keras`

These files preserve trained model parameters and configurations for later evaluation and inference.

The exact relationship between each checkpoint and the corresponding experiment should be verified using the training scripts and saved experiment metadata.

## 12. Segmentation Implementation

The segmentation implementation is located in the `src/` directory.

Relevant files include:

| File | Purpose |
|---|---|
| `src/models/unet.py` | Defines the U-Net architecture and segmentation-related functions. |
| `src/training/train_unet.py` | Implements baseline U-Net training. |
| `src/training/train_unet_flip_only.py` | Implements flip-only augmentation training. |
| `src/training/train_unet_augmented.py` | Implements the augmented U-Net training experiment. |
| `src/evaluation/evaluate_unet.py` | Evaluates trained segmentation models. |

The evaluation pipeline also produces segmentation metrics, visualizations, and comparison outputs.

## 13. Segmentation Visualizations

Visual inspection is important because numerical metrics cannot fully describe the quality of a predicted tumor mask.

The project includes visualizations comparing:

1. Original MRI image.
2. Ground-truth segmentation mask.
3. Predicted segmentation mask.
4. Overlay of the predicted tumor region on the MRI image.

These examples help identify:

- Correctly segmented tumor regions.
- Missed tumor areas.
- Incorrectly predicted regions.
- Boundary inaccuracies.
- Differences between segmentation experiments.

**Photo location:**

`docs/photos/segmentation/`

## 14. Segmentation Evaluation

Segmentation models are evaluated using overlap-based and boundary-related metrics.

The Dice coefficient measures overlap between predicted and ground-truth tumor regions.

Intersection over Union (IoU) measures the ratio between the intersection and union of the two regions.

Boundary-related metrics provide additional information about the accuracy of predicted tumor boundaries.

The detailed definitions, formulas, and evaluation procedures are described in:

`07_evaluation_metrics.md`

## 15. Limitations of U-Net Segmentation

Although U-Net is effective for many biomedical segmentation tasks, several limitations must be considered.

### 15.1 Limited Dataset Size

The available dataset contains a limited number of patients.

This can restrict the diversity of anatomical and tumor characteristics represented during training.

### 15.2 Small or Difficult Tumor Regions

Small tumor regions may be difficult to identify accurately.

A relatively small boundary error can significantly affect overlap-based segmentation metrics when the tumor region occupies only a small part of the image.

### 15.3 Boundary Accuracy

A model may achieve a reasonable Dice score while still producing inaccurate tumor boundaries.

For this reason, boundary-related evaluation is included in the research.

### 15.4 Generalization

Segmentation performance on the available dataset does not guarantee equivalent performance on MRI images from different institutions, scanners, or patient populations.

### 15.5 Augmentation Limitations

Data augmentation can introduce useful image variations, but its effectiveness depends on the transformations used and whether they preserve meaningful anatomical characteristics.

## 16. Chapter Summary

This chapter described the segmentation component of BrainTumorResearch.

The project uses a U-Net-based architecture to predict tumor regions in brain MRI images.

The U-Net architecture combines an encoder, bottleneck, decoder, and skip connections to capture contextual and spatial information.

The research investigates three training configurations:

- Baseline U-Net.
- Flip-only U-Net.
- Flip and rotation U-Net.

The recorded image-level mean Dice scores are 0.7016, 0.7259, and 0.7281, respectively.

The project also includes patient-level evaluation, boundary-related metrics, visual comparisons, and statistical analysis to provide a more complete understanding of segmentation performance.

The next chapter describes the training procedures and experiments used to develop and compare the classification and segmentation models.

---

**Document:** 05 — Segmentation Models  
**Project:** BrainTumorResearch  
**Architecture:** U-Net  
**Task:** Brain Tumor MRI Segmentation  
**Experiments:** Baseline, Flip-only, Flip and Rotation
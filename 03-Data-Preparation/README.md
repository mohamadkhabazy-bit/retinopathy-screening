# Data Preparation

## Overview

This phase prepares the retinal fundus images for deep learning by applying a sequence of preprocessing, augmentation, normalization, and sampling techniques specifically designed for medical image analysis.

The objective is to improve image quality, preserve retinal anatomy, increase data diversity, and reduce the impact of class imbalance.

---

## Ben Graham Preprocessing

Each retinal image is first processed using the **Ben Graham preprocessing** technique.

This method applies Gaussian blurring followed by weighted image subtraction to suppress uneven illumination while enhancing retinal structures such as blood vessels, microaneurysms, hemorrhages, and exudates. This step reduces acquisition-related variability caused by different fundus cameras and imaging conditions.

---

## Green Channel Extraction

After preprocessing, the **green channel** is extracted and replicated into a three-channel image.

Among the RGB channels, the green channel provides the highest contrast for retinal vessels and diabetic retinopathy lesions, allowing the network to learn clinically relevant features more effectively while discarding less informative color information.

---

## Preserving Retinal Geometry

Instead of directly resizing images to a fixed resolution, the preprocessing pipeline uses:

- **LongestMaxSize**
- **PadIfNeeded**

This strategy preserves the original aspect ratio of the retina while padding the image to **512 × 512** pixels.

Maintaining anatomical geometry prevents distortion of retinal vessels, the optic disc, the macula, and pathological regions, ensuring that clinically meaningful structures remain geometrically consistent.

---

## Data Augmentation

Different augmentation policies are applied during training to improve model generalization while preserving medically meaningful retinal structures.

### Standard Augmentations (Majority Classes)

The majority class (**No DR** and **Moderate**) uses a conservative augmentation pipeline consisting of:

- Horizontal Flip
- Vertical Flip
- Random 90° Rotation
- Random Brightness & Contrast

<p align="center">
  <img src="visualizations/pipeline_majority.png" width="85%">
</p>

### Enhanced Augmentations (Minority Classes)

To compensate for the limited number of samples in the **Mild**, **Severe**, and **Proliferative DR** classes, additional augmentations are applied:

- ShiftScaleRotate
- CLAHE (Contrast Limited Adaptive Histogram Equalization)

These transformations increase sample diversity while preserving retinal anatomy and improving the visibility of subtle lesions.

<p align="center">
  <img src="visualizations/pipeline_minority.png" width="85%">
</p>

---

## CLAHE Enhancement

CLAHE is selectively applied to minority classes to improve local contrast without over-amplifying image noise. This enhances small retinal lesions such as microaneurysms and hemorrhages, making them more distinguishable for the neural network.

<p align="center">
  <img src="visualizations/clahe_effect.png" width="80%">
</p>

---

## Anatomy-Aware Augmentation

Several common computer vision augmentations were intentionally excluded because they may damage retinal anatomy.

The following transformations were removed:

- GridDistortion
- ElasticTransform
- CoarseDropout

These operations can distort blood vessels or remove tiny lesions such as microaneurysms, leading to unrealistic medical images and degraded model performance.

---

## Class Imbalance Handling

The APTOS dataset exhibits severe class imbalance.

To address this issue, two complementary strategies were adopted.

### Weighted Random Sampling

A **WeightedRandomSampler** is used during training to generate balanced mini-batches and increase the frequency of minority-class samples.

### Softened Class Weights

Class weights are computed using Scikit-learn's `compute_class_weight()` function.

Instead of using the original weights directly, their square root is applied before training.

This soft-weighting strategy reduces the risk of **double penalty**, where minority classes would otherwise receive excessive emphasis through both sampling and loss weighting.

---

## Normalization

Finally, all images are normalized using the standard **ImageNet** mean and standard deviation.

This normalization ensures compatibility with the pretrained EfficientNet backbone, leading to faster convergence, more stable optimization, and improved transfer learning performance.

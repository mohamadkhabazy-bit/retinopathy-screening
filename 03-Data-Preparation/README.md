# Data Preparation

## Overview

This phase prepares the retinal fundus images for deep learning by applying a sequence of preprocessing, augmentation, normalization, and sampling techniques specifically designed for medical image analysis.

The objective is to improve image quality, preserve retinal anatomy, increase data diversity, and reduce the impact of class imbalance.

---

## Ben Graham Preprocessing

Each retinal image is first processed using the **Ben Graham preprocessing** technique.

This method applies Gaussian blurring followed by weighted image subtraction to reduce illumination variations while enhancing important retinal structures such as blood vessels, microaneurysms, and hemorrhages.

---

## Green Channel Extraction

After preprocessing, the **green channel** is extracted and replicated into a three-channel image.

Among the RGB channels, the green channel provides the highest contrast for retinal vessels and diabetic retinopathy lesions, allowing the network to learn discriminative features more effectively.

---

## Preserving Retinal Geometry

Instead of directly resizing images to a fixed resolution, the preprocessing pipeline uses:

- **LongestMaxSize**
- **PadIfNeeded**

This strategy preserves the original aspect ratio of the retina while padding the image to **512 × 512** pixels.

Maintaining anatomical geometry prevents distortion of retinal vessels and pathological regions.

---

## Data Augmentation

Different augmentation policies are applied during training to improve model generalization.

### Standard Augmentations

- Horizontal Flip
- Vertical Flip
- Random 90° Rotation
- Random Brightness and Contrast

### Minority-Class Augmentations

For minority classes (Mild, Severe, and Proliferative), additional augmentation is applied:

- ShiftScaleRotate
- CLAHE (Contrast Limited Adaptive Histogram Equalization)

These augmentations improve feature diversity while preserving medically meaningful structures.

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

This soft-weighting strategy reduces the risk of **Double Penalty**, where minority classes would otherwise receive excessive emphasis through both sampling and loss weighting.

---

## Normalization

Finally, all images are normalized using the standard **ImageNet** mean and standard deviation.

This normalization ensures compatibility with the pretrained EfficientNet backbone and contributes to faster convergence and more stable optimization.

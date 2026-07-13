# Data Understanding

## Overview

This phase focuses on understanding the characteristics of the **APTOS 2019 Blindness Detection** dataset before any preprocessing or model development.

---

## Dataset

The project uses the **APTOS 2019 Blindness Detection** dataset, which contains color retinal fundus images collected from diabetic patients.

- **Training images:** 3,662
- **Image type:** RGB fundus photographs
- **Classification task:** 5-class ordinal classification

| Grade | Description | Training Images |
|:-----:|-------------|----------------:|
| 0 | No DR | 1,805 |
| 1 | Mild | 370 |
| 2 | Moderate | 999 |
| 3 | Severe | 193 |
| 4 | Proliferative DR | 295 |
| **Total** | | **3,662** |

---

## Problem Characteristics

Diabetic Retinopathy grading is an **ordinal classification** problem rather than a conventional multi-class classification task. Misclassifying neighboring grades is clinically less severe than predicting completely different grades.

**Examples**

- Mild → Moderate ✔️ Acceptable
- Severe → Proliferative ✔️ Acceptable
- No DR → Proliferative ❌ Serious error

This characteristic strongly influenced the design of the loss function and evaluation metric.

---

## Dataset Challenges

Several challenges were identified during dataset exploration:

- Significant class imbalance
- Limited number of samples for advanced DR stages
- High intra-class variability
- Small retinal lesions occupying only a tiny portion of each image
- Variations in illumination, focus, and image acquisition conditions

---

## Class Distribution

The dataset is highly imbalanced, with the majority of samples belonging to the **No DR** class, while **Mild**, **Severe**, and **Proliferative DR** contain considerably fewer images.

This imbalance required specialized sampling and weighting strategies during model training.

<p align="center">
  <img src="../Data%20Preparation/visualizations/class_distribution.png" width="70%">
</p>

---

## Sample Images

Representative retinal fundus images from different disease grades illustrate the substantial visual differences in lesion severity and retinal appearance.

<p align="center">
  <img src="../Data%20Preparation/visualizations/sample_images.png" width="90%">
</p>

---

## Image Characteristics

Understanding the image properties is essential for designing an effective preprocessing pipeline.

### Image Quality Analysis

The dataset contains noticeable variations in illumination, sharpness, and overall image quality due to differences in acquisition devices and imaging conditions.

<p align="center">
  <img src="../Data%20Preparation/visualizations/image_quality_analysis.png" width="80%">
</p>

### Image Dimensions

Although the original images have varying resolutions, they maintain a similar aspect ratio. Therefore, all images were resized to a fixed input size during preprocessing to ensure consistent model training.

<p align="center">
  <img src="../Data%20Preparation/visualizations/image_dimensions.png" width="80%">
</p>

---

## Key Findings

The exploratory analysis revealed several important observations:

- Retinal lesions are extremely small compared to the entire image.
- Preserving anatomical structures is critical.
- Standard computer vision augmentations may distort medically important features.
- The ordinal relationship between disease grades must be considered during training and evaluation.

These findings directly guided the design of the preprocessing pipeline, augmentation strategy, model architecture, loss function, and evaluation methodology.

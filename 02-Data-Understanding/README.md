# Data Understanding

## Overview

This phase focuses on understanding the characteristics of the **APTOS 2019 Blindness Detection** dataset before any preprocessing or model development. A comprehensive exploratory analysis was conducted to examine the dataset composition, image quality, and imaging characteristics, providing the foundation for designing an effective preprocessing pipeline and training strategy.

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
- Variations in illumination, contrast, focus, and image acquisition conditions
- Images captured using different fundus cameras with varying resolutions

---

## Class Distribution

The dataset is highly imbalanced, with the majority of samples belonging to the **No DR** class, while **Mild**, **Severe**, and **Proliferative DR** contain considerably fewer images.

This imbalance increases the risk of model bias toward majority classes and motivated the use of specialized sampling and class-balanced loss weighting during training.

<p align="center">
  <img src="../03-Data-Preparation/visualizations/class_distribution.png" width="75%">
</p>

---

## Sample Images

Representative retinal fundus images from different disease grades illustrate the progressive appearance of retinal lesions as disease severity increases. While advanced stages contain more visible pathological features, many early-stage abnormalities occupy only a very small portion of the retinal image, making automatic detection particularly challenging.

<p align="center">
  <img src="../03-Data-Preparation/visualizations/sample_images.png" width="90%">
</p>

---

## Image Quality Analysis

The image quality analysis reveals noticeable variations in both brightness and contrast across the five DR grades. Severe images exhibit the highest average brightness, whereas Proliferative and No DR images are comparatively darker. Similarly, contrast varies between classes, with No DR showing the highest average contrast and Proliferative images the lowest.

Moreover, the relatively large standard deviations indicate considerable variability in imaging conditions even within the same class. These differences are primarily attributed to variations in fundus cameras, illumination settings, and acquisition protocols rather than disease severity alone.

These observations motivated the adoption of the **Ben Graham preprocessing** technique to suppress uneven illumination and normalize image appearance across samples. In addition, **green channel extraction** was employed because retinal blood vessels and diabetic lesions exhibit the highest contrast in the green channel, making clinically relevant structures more distinguishable while reducing the influence of less informative color information.

Together, these preprocessing steps improve image consistency and enhance lesion visibility, providing more informative inputs for the subsequent deep learning model.

<p align="center">
  <img src="../03-Data-Preparation/visualizations/image_quality_analysis.png" width="80%">
</p>

---

## Image Dimensions

The image dimension analysis shows substantial variability in image resolution, with widths ranging from **474** to **4288** pixels and heights from **358** to **2848** pixels. Such large differences make it impractical to feed the original images directly into a deep learning model, necessitating a unified input size.

Despite these resolution differences, the aspect ratio remains relatively consistent across the dataset (**1.28 ± 0.18**), indicating that the anatomical structure of retinal fundus images is largely preserved.

These observations motivated the use of an aspect-ratio-preserving resizing strategy. Each image is first resized using **LongestMaxSize**, ensuring that the longer side matches the target input size while maintaining the original geometric proportions. The remaining space is then filled using **PadIfNeeded**, producing a square input without introducing geometric distortion.

Preserving the original aspect ratio is particularly important in medical imaging, as stretching or compressing retinal images can distort anatomical structures such as blood vessels, the optic disc, the macula, and retinal lesions. Such distortions may alter clinically meaningful features and negatively affect both model learning and diagnostic reliability.

<p align="center">
  <img src="../03-Data-Preparation/visualizations/image_dimensions.png" width="80%">
</p>

---

## Key Findings

The exploratory analysis revealed several important observations:

- The dataset exhibits a severe class imbalance, requiring class-balanced sampling and weighted optimization.
- Retinal lesions occupy only a very small portion of the fundus image, demanding high-quality feature preservation.
- Significant variations in brightness and contrast justify illumination normalization and contrast enhancement during preprocessing.
- The green channel provides superior visibility of retinal vessels and pathological lesions compared to the other color channels.
- Images vary substantially in resolution while maintaining a relatively consistent aspect ratio, making aspect-ratio-preserving resizing preferable to direct image scaling.
- Preserving anatomical structures is essential, as geometric distortion may compromise clinically relevant information.
- The ordinal relationship between disease grades should be considered during both optimization and evaluation.

These findings directly guided the design of the preprocessing pipeline, including **Ben Graham preprocessing**, **green channel extraction**, **CLAHE**, **aspect-ratio-preserving resizing**, and **padding**, as well as the subsequent model architecture, training strategy, loss function, and evaluation methodology.

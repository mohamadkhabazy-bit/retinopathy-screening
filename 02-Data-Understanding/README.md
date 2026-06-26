# Data Understanding

## Overview

This phase focuses on understanding the characteristics of the APTOS 2019 Blindness Detection dataset before any preprocessing or model development.

## Dataset

The project uses the **APTOS 2019 Blindness Detection** dataset, which contains color retinal fundus images collected from diabetic patients.

- Total images: **3,662**
- Image type: RGB Fundus photographs
- Classification task: 5-class ordinal classification

| Grade | Description |
|------:|-------------|
| 0 | No DR |
| 1 | Mild |
| 2 | Moderate |
| 3 | Severe |
| 4 | Proliferative DR |

## Problem Characteristics

Diabetic Retinopathy grading is an **ordinal classification** problem rather than a conventional multi-class classification task. Misclassifying neighboring grades is clinically less severe than predicting completely different grades.

Examples:

- Mild → Moderate ✔️ Acceptable
- Severe → Proliferative ✔️ Acceptable
- No DR → Proliferative ❌ Serious error

This characteristic strongly influenced the design of the loss function and evaluation metric.

## Dataset Challenges

Several challenges were identified during dataset exploration:

- Significant class imbalance
- Limited number of samples for advanced DR stages
- High intra-class variability
- Small retinal lesions occupying only a tiny portion of each image
- Variations in illumination, focus, and image acquisition conditions

## Class Distribution

The dataset is highly imbalanced, with the majority of samples belonging to the **No DR** class, while **Mild**, **Severe**, and **Proliferative** contain considerably fewer images.

This imbalance required specialized sampling and weighting strategies during model training.

## Key Findings

The exploratory analysis revealed that:

- Retinal lesions are extremely small compared to the entire image.
- Preserving anatomical structure is critical.
- Standard computer vision augmentations may distort medically important features.
- The ordinal relationship between disease grades must be considered during training and evaluation.

These observations directly guided the design of the preprocessing pipeline, augmentation strategy, model architecture, loss function, and evaluation methodology.

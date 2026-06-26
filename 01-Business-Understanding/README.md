# Business Understanding

## Overview

This phase defines the business objectives, project scope, and success criteria following the CRISP-DM methodology.

The goal of this project is to develop an artificial intelligence system capable of automatically detecting and grading **Diabetic Retinopathy (DR)** from retinal fundus images. The system is designed to assist ophthalmologists by providing fast, accurate, and reliable screening while reducing diagnostic workload.

---

## Business Problem

Diabetic Retinopathy is one of the leading causes of preventable blindness worldwide. Early diagnosis can significantly reduce the risk of severe vision loss; however, manual examination of retinal images is time-consuming and requires experienced specialists.

An automated deep learning solution can support clinical decision-making by providing consistent and scalable screening.

---

## Project Objectives

- Develop a deep learning model for Diabetic Retinopathy grading.
- Classify retinal fundus images into five disease severity levels.
- Achieve high clinical reliability using an ordinal-aware evaluation metric.
- Reduce misclassification between distant severity grades.
- Build a reproducible machine learning pipeline following the CRISP-DM framework.

---

## Success Criteria

The project is considered successful if it meets the following objectives:

- High Quadratic Weighted Kappa (QWK)
- High classification accuracy
- Robust performance on minority classes
- Stable training without severe overfitting
- Efficient execution on consumer-grade GPUs

---

## Dataset

- **Dataset:** APTOS 2019 Blindness Detection
- **Image Type:** Retinal Fundus Images
- **Number of Classes:** 5
- **Classification Type:** Ordinal Multi-Class Classification

Classes:

| Label | Description |
|------:|-------------|
| 0 | No DR |
| 1 | Mild |
| 2 | Moderate |
| 3 | Severe |
| 4 | Proliferative DR |

---

## Repository Structure

This directory contains documentation related to the Business Understanding phase of the project.

```
Business Understanding/
└── README.md
```

---

## CRISP-DM Phase

This directory corresponds to the **Business Understanding** phase of the CRISP-DM methodology.

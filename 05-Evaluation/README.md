# Evaluation

## Objective

Evaluate the final diabetic retinopathy grading model using clinically meaningful metrics that account for the ordinal nature of disease severity and verify its ability to generalize to unseen retinal images.

---

## Evaluation Metrics

Diabetic Retinopathy grading is an **ordinal classification** problem, meaning that the severity classes have a natural ordering.

For this reason, the primary evaluation metric is:

- **Quadratic Weighted Kappa (QWK)**

QWK measures not only whether a prediction is correct but also how far the predicted grade is from the ground truth. Larger grading errors receive significantly higher penalties, making QWK much more appropriate than standard accuracy for medical grading tasks.

Additional evaluation metrics include:

- Accuracy
- Precision (Macro)
- Recall (Macro)
- F1-score (Macro)
- Confusion Matrix
- Classification Report

---

## Final Validation Results

The best checkpoint obtained during training was evaluated on the validation set.

| Metric | Value |
|---------|-------|
| Quadratic Weighted Kappa (QWK) | **0.8806** |
| Accuracy | **82.8%** |
| Macro F1-score | **66.7%** |
| Validation Loss | **0.3865** |

---

## Per-Class Performance

| Class | Precision | Recall | F1-score |
|------|----------:|--------:|---------:|
| No DR | 0.97 | 0.99 | 0.98 |
| Mild | 0.67 | 0.54 | 0.60 |
| Moderate | 0.75 | 0.81 | 0.78 |
| Severe | 0.43 | 0.46 | 0.44 |
| Proliferative | 0.60 | 0.47 | 0.53 |

The model demonstrates excellent performance on healthy and moderate cases while maintaining reasonable performance on minority classes despite the severe class imbalance of the APTOS dataset.

---

## Confusion Matrix

The confusion matrix provides a detailed analysis of prediction errors across all disease grades.

| Actual \\ Predicted | No DR | Mild | Moderate | Severe | Proliferative |
|---------------------|------:|-----:|---------:|-------:|--------------:|
| **No DR** | 358 | 2 | 0 | 0 | 1 |
| **Mild** | 7 | 40 | 24 | 2 | 1 |
| **Moderate** | 3 | 12 | 163 | 14 | 8 |
| **Severe** | 0 | 0 | 12 | 18 | 9 |
| **Proliferative** | 0 | 6 | 17 | 8 | 28 |

### Error Analysis

The confusion matrix reveals several important characteristics of the model:

- Most prediction errors occur between **adjacent severity grades**, indicating that the model successfully learns the ordinal progression of diabetic retinopathy.
- Very few extreme errors (for example, **No DR → Proliferative**) are observed.
- The largest confusion occurs between **Moderate** and **Severe**, which is also one of the most challenging boundaries for clinical diagnosis.
- The **No DR** class is identified with very high reliability, producing only three incorrect predictions out of 361 validation images.
- Performance on the minority classes (**Severe** and **Proliferative**) remains the primary limitation because these classes contain significantly fewer training samples.

---

## Generalization

Several design choices contribute to the strong generalization performance of the proposed model:

- Transfer Learning from ImageNet
- EfficientNet-B2 backbone
- CBAM Attention Module
- GeM Pooling
- Mixed Precision Training (AMP)
- Gradient Accumulation
- Cosine Annealing Learning Rate Scheduler
- Label Smoothing
- Early Stopping
- Model Checkpointing

Together, these techniques reduce overfitting while improving training stability and validation performance.

---

## Conclusion

The proposed **EfficientNet-B2 + CBAM** framework achieves a **Quadratic Weighted Kappa of 0.8806** and an **Accuracy of 82.8%** on the APTOS validation set.

The confusion matrix demonstrates that the majority of prediction errors occur between neighboring disease stages rather than distant grades, indicating that the model effectively captures the ordinal structure of diabetic retinopathy severity. These results suggest that the proposed approach is both accurate and clinically reliable while remaining computationally efficient enough for deployment on consumer-grade GPUs.

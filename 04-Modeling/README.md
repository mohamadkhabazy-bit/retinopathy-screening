# Modeling

## Objective
Develop a deep learning model capable of automatically grading diabetic retinopathy severity from retinal fundus images while preserving high clinical reliability and computational efficiency.

---

## Model Architecture

The final model is based on **EfficientNet-B2** with transfer learning from ImageNet.

The architecture consists of four main components:

- EfficientNet-B2 Backbone
- CBAM Attention Module
- GeM Pooling
- Classification Head

The backbone extracts high-level retinal features, while the CBAM module improves feature representation by applying both channel and spatial attention. Instead of Global Average Pooling, GeM Pooling is used to better preserve discriminative lesion information before classification.

The classification head contains:

- Fully Connected layer
- Batch Normalization
- ReLU activation
- Dropout
- Final Linear classifier (5 classes)

---

## Training Strategy

Training is performed using a two-phase fine-tuning strategy.

### Phase 1

- Backbone frozen
- Only CBAM and classification head are trained
- BatchNorm statistics remain frozen to preserve ImageNet representations

### Phase 2

- Final EfficientNet-B2 block is unfrozen
- Surgical fine-tuning with discriminative learning rates
- Warm-up is applied before full optimization

---

## Loss Function

The problem is treated as an ordinal classification task.

A hybrid loss function is used:

- 70% Earth Mover's Distance (EMD) Loss
- 30% Cross Entropy Loss

Cross Entropy employs Label Smoothing (0.1) to reduce overconfidence and improve generalization.

---

## Optimization

Training uses:

- AdamW optimizer
- Cosine Annealing Learning Rate Scheduler
- Learning Rate Warmup

---

## Training Stabilization

Several techniques are incorporated to improve stability and generalization:

- Mixed Precision Training (AMP)
- Gradient Accumulation
- Gradient Clipping
- Early Stopping
- Model Checkpointing

---

## Output

The model predicts one of five diabetic retinopathy grades:

- No DR
- Mild
- Moderate
- Severe
- Proliferative DR

The trained model is stored in the `checkpoints/` directory.

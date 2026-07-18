# 4. Modeling

The modeling phase focuses on designing a robust deep learning architecture capable of accurately grading diabetic retinopathy while preserving the ordinal relationship between disease severity levels. The proposed framework combines transfer learning, attention mechanisms, ordinal-aware optimization, and a multi-stage fine-tuning strategy to maximize the Quadratic Weighted Kappa (QWK) score.

---

## 4.1 Model Architecture

The proposed model consists of four major components:

```
Input Image
      │
      ▼
EfficientNet-B4 Backbone
      │
      ▼
CBAM Attention Module
      │
      ▼
GeM Pooling
      │
      ▼
Classification Head
      │
      ▼
Class Probabilities
      │
      ▼
Expected Grade
      │
      ▼
Ordinal Thresholds
      │
      ▼
Final DR Grade
```

---

## 4.2 Backbone Network

The backbone of the network is **EfficientNet-B4**, initialized with ImageNet pretrained weights using the `timm` library.

Unlike conventional CNN architectures, EfficientNet scales network depth, width, and input resolution simultaneously using compound scaling, providing an excellent balance between accuracy and computational efficiency.

The classification layer of the pretrained model is removed (`num_classes=0`) so that the network acts purely as a feature extractor.

Key characteristics include:

- ImageNet pretrained initialization
- Compound scaling architecture
- High parameter efficiency
- Strong transfer learning capability
- 1792-dimensional feature representation

---

## 4.3 CBAM Attention Module

To enhance lesion localization, a **Convolutional Block Attention Module (CBAM)** is placed immediately after the EfficientNet backbone.

CBAM sequentially applies two complementary attention mechanisms.

### Channel Attention

Channel attention learns which feature channels contain the most clinically relevant information.

The attention weights are generated using:

- Global Average Pooling
- Global Max Pooling
- Shared Multi-Layer Perceptron (MLP)
- Sigmoid activation

This enables the model to emphasize channels containing retinal lesions while suppressing less informative responses.

---

### Spatial Attention

After channel refinement, spatial attention determines **where** important pathological patterns are located within the retinal image.

Spatial attention is computed by combining

- channel-wise average projection
- channel-wise max projection

followed by a **7×7 convolution** and sigmoid activation.

This mechanism allows the network to focus on lesion regions such as

- microaneurysms
- hemorrhages
- exudates
- neovascularization

instead of background retinal structures.

---

## 4.4 Generalized Mean Pooling (GeM)

Instead of Global Average Pooling (GAP), the model employs **Generalized Mean Pooling (GeM)**.

GeM introduces a learnable exponent \(p\), enabling the pooling operation to smoothly interpolate between average pooling and max pooling.

The pooling operation is defined as

$$
\mathrm{GeM}(X)=
\left(
\frac{1}{N}
\sum_{i=1}^{N}
x_i^{\,p}
\right)^{\frac{1}{p}}
$$

where the exponent \(p\) is optimized during training.

Compared with conventional average pooling, GeM preserves localized lesion information more effectively, making it particularly suitable for medical image analysis where abnormalities often occupy only a small portion of the image.

---

## 4.5 Classification Head

Following GeM pooling, the extracted features are passed through a lightweight fully connected classifier.

Architecture:

```text
Flatten
    ↓
Linear (1792 → 1024)
    ↓
Batch Normalization
    ↓
ReLU
    ↓
Dropout (0.4)
    ↓
Linear (1024 → 5)
```

The classifier predicts the probability distribution over the five diabetic retinopathy grades.

Dropout is used to reduce overfitting, while Batch Normalization stabilizes optimization during training.

---

## 4.6 Loss Function

Diabetic retinopathy grading is inherently an **ordinal classification problem**.

Misclassifying Grade 4 as Grade 3 is clinically less severe than predicting Grade 4 as Grade 0.

To incorporate this property, the model optimizes a hybrid loss function composed of Earth Mover's Distance (EMD) loss and Cross Entropy loss.

### Earth Mover's Distance (EMD)

The EMD loss compares the cumulative probability distributions of the predicted and target labels.

Unlike Cross Entropy, EMD explicitly penalizes predictions according to their ordinal distance.

Consequently, predictions that are closer to the true class receive a much smaller penalty.

---

### Cross Entropy Loss

Cross Entropy provides stable optimization during training while encouraging accurate probability estimation.

Label smoothing with a smoothing factor of **0.05** is applied to improve calibration and reduce overconfidence.

---

### Combined Objective

The final optimization objective is

$$
L = \alpha L_{\mathrm{EMD}} + (1-\alpha)L_{\mathrm{CE}}
$$

where

- α = 0.7

This weighting places greater emphasis on ordinal consistency while retaining the optimization stability of Cross Entropy.

---

## 4.7 Ordinal Decision Rule

Unlike conventional classifiers, the final prediction is **not obtained using the argmax operator**.

Instead, the probability distribution predicted by the network is converted into a continuous severity score using the expected value

$$
\hat{y} = \sum_{k=0}^{4} k\,P(k)
$$

where \(P(k)\) represents the predicted probability of class \(k\).

This continuous score is subsequently converted into one of the five diabetic retinopathy grades using four decision thresholds.

This ordinal decision rule is significantly better aligned with Quadratic Weighted Kappa than standard argmax classification.

---

## 4.8 Threshold Optimization

The four decision thresholds are optimized on the validation set using coordinate ascent.

Rather than optimizing the neural network itself, this procedure searches for threshold values that maximize the Quadratic Weighted Kappa score.

The optimized thresholds are saved and reused during inference to ensure consistency between validation and deployment.

---

## 4.9 Training Strategy

Transfer learning is performed progressively in two stages.

### Phase 1 – Feature Learning

During the first stage,

- the EfficientNet backbone remains completely frozen;
- only the CBAM module and the classification head are optimized.

This allows the newly added layers to adapt to retinal images while preserving the pretrained ImageNet representations.

Training configuration:

- Optimizer: AdamW
- Learning Rate: **3×10⁻⁴**
- Weight Decay: **1×10⁻⁴**
- Scheduler: ReduceLROnPlateau
- Batch Size: 32
- Gradient Accumulation: 2
- Effective Batch Size: 64

---

### Phase 2 – Progressive Fine-Tuning

After convergence of the classifier, the final EfficientNet stage is unfrozen for fine-tuning.

Different learning rates are assigned to pretrained and newly initialized parameters.

| Component | Learning Rate |
|-----------|--------------:|
| Backbone | 5×10⁻⁵ |
| CBAM + Head | 1×10⁻⁴ |

The scheduler is changed to **Cosine Annealing** with a five-epoch linear warmup.

This strategy enables gradual adaptation of pretrained features while preventing catastrophic forgetting.

---

## 4.10 Batch Normalization Strategy

During fine-tuning, Batch Normalization statistics are updated only for trainable backbone layers.

BatchNorm layers belonging to frozen blocks remain in evaluation mode, preserving their pretrained running statistics.

This selective freezing stabilizes optimization while allowing genuine fine-tuning of the unfrozen layers.

---

## 4.11 Optimization

The model is optimized using **AdamW**.

To improve training stability and computational efficiency, several optimization techniques are employed:

- Automatic Mixed Precision (AMP)
- Gradient Accumulation
- Gradient Clipping (maximum norm = 1.0)
- Automatic Checkpointing
- Resume Training
- Early Stopping based on validation QWK

---

## 4.12 Model Selection

The validation **Quadratic Weighted Kappa (QWK)** serves as the primary criterion for model selection.

Whenever the validation QWK improves,

- the model weights,
- optimizer state,
- scheduler state,
- current epoch,
- and best validation score

are automatically saved.

Training terminates early if no improvement is observed for a predefined number of epochs, preventing unnecessary computation and reducing overfitting.

---

## 4.13 Hyperparameter Configuration

The following hyperparameters were selected empirically to achieve a balance between convergence speed, generalization performance, and GPU memory efficiency.

### Phase 1 – Feature Learning

| Hyperparameter | Value |
|---------------|------:|
| Backbone | EfficientNet-B4 |
| Input Resolution | 512 × 512 |
| Batch Size | 32 |
| Gradient Accumulation | 2 |
| Effective Batch Size | 64 |
| Optimizer | AdamW |
| Initial Learning Rate | 3 × 10⁻⁴ |
| Weight Decay | 1 × 10⁻⁴ |
| Scheduler | ReduceLROnPlateau |
| Epochs | 15 |
| Early Stopping Patience | 10 |
| Dropout | 0.4 |
| Label Smoothing | 0.05 |
| α (Combined Loss) | 0.7 |
| Gradient Clipping | 1.0 |
| Mixed Precision | Enabled |
| Frozen Backbone | Yes |
| Trainable Layers | CBAM + Classification Head |

---

### Phase 2 – Progressive Fine-Tuning

| Hyperparameter | Value |
|---------------|------:|
| Unfrozen Backbone Stages | Last Stage |
| Backbone Learning Rate | 5 × 10⁻⁵ |
| Head Learning Rate | 1 × 10⁻⁴ |
| Optimizer | AdamW |
| Weight Decay | 1 × 10⁻⁴ |
| Scheduler | Cosine Annealing |
| Warmup Epochs | 5 |
| Maximum Epochs | 25 |
| Early Stopping Patience | 15 |
| Batch Size | 32 |
| Gradient Accumulation | 2 |
| Effective Batch Size | 64 |
| Mixed Precision | Enabled |
| BatchNorm Strategy | Frozen on frozen layers |
| Gradient Clipping | 1.0 |
| Test-Time Augmentation | Enabled |

---

### Ordinal Learning Parameters

| Parameter | Value |
|-----------|------:|
| Number of Classes | 5 |
| Decision Rule | Expected Value |
| Thresholds | Validation Optimized |
| Threshold Optimization | Coordinate Ascent |
| Primary Metric | Quadratic Weighted Kappa |
| Final Evaluation | TTA + Ordinal Thresholds |

---

## 4.14 Design Choices

The proposed architecture was intentionally designed so that each component addresses a specific limitation commonly encountered in diabetic retinopathy grading.

---

### Why EfficientNet-B4?

EfficientNet-B4 provides an excellent compromise between representational capacity and computational efficiency.

Compared with smaller variants (e.g., B0 or B2), B4 captures richer retinal structures while remaining trainable on modern GPUs using mixed precision and gradient accumulation.

Its compound scaling strategy enables simultaneous scaling of network depth, width, and input resolution, leading to stronger feature extraction without excessive computational cost.

---

### Why CBAM?

Retinal lesions generally occupy only a small portion of the fundus image.

Without attention mechanisms, CNNs often allocate unnecessary capacity to healthy retinal regions.

CBAM improves feature representation through two complementary mechanisms:

- **Channel Attention**, which identifies *what* features are important.
- **Spatial Attention**, which identifies *where* important pathological regions are located.

This enables the network to concentrate on clinically relevant structures such as microaneurysms, hemorrhages, hard exudates, cotton-wool spots, and neovascularization.

---

### Why GeM Pooling?

Traditional Global Average Pooling treats every spatial location equally.

However, diabetic retinopathy lesions are typically sparse and localized.

GeM introduces a learnable pooling exponent that adaptively interpolates between average pooling and max pooling.

Consequently, localized pathological responses contribute more strongly to the final feature representation while preserving global contextual information.

---

### Why a Combined Loss?

Cross Entropy assumes that all classification errors are equally severe.

This assumption does not hold for diabetic retinopathy grading, where disease severity follows a natural clinical order.

Earth Mover's Distance (EMD) explicitly models this ordinal relationship by comparing cumulative probability distributions.

However, EMD alone may lead to slower optimization and less stable convergence.

Combining EMD with Cross Entropy provides both

- ordinal awareness,
- and stable probabilistic optimization.

---

### Why Expected Value Instead of Argmax?

Conventional classifiers obtain predictions using

```text
Predicted Class = argmax(P)
```

Although appropriate for nominal classification problems, argmax ignores the ordinal relationship between diabetic retinopathy grades.

Instead, this project computes a continuous severity score as the expected value of the predicted probability distribution

\[
\hat{y}=\sum_{k=0}^{4}kP(k)
\]

The continuous score is subsequently converted into one of the five grades using optimized decision thresholds.

This approach better preserves disease ordering and aligns naturally with the Quadratic Weighted Kappa metric.

---

### Why Threshold Optimization?

Even when probability estimation is accurate, fixed decision boundaries are not necessarily optimal for maximizing Quadratic Weighted Kappa.

Therefore, four decision thresholds are optimized on the validation set through coordinate ascent.

Only the thresholds are optimized—the neural network parameters remain unchanged.

This lightweight post-processing step improves agreement with expert annotations without requiring additional model training.

---

### Why Two-Stage Training?

Training the entire EfficientNet backbone from the beginning may destabilize pretrained ImageNet representations.

To avoid catastrophic forgetting, training is performed progressively.

During **Phase 1**, only the newly initialized layers (CBAM and the classification head) are optimized while the backbone remains frozen.

After convergence, **Phase 2** unfreezes only the final EfficientNet stage using a substantially smaller learning rate.

This gradual adaptation preserves generic visual features while specializing higher-level representations for retinal pathology.

---

### Why Differential Learning Rates?

Pretrained layers already contain useful visual representations.

Applying the same learning rate to both pretrained and newly initialized layers often damages these representations.

Therefore,

- pretrained backbone layers receive a smaller learning rate,
- whereas CBAM and the classification head receive a larger learning rate.

This strategy enables efficient fine-tuning while minimizing catastrophic forgetting.

---

### Why Freeze Batch Normalization?

Updating Batch Normalization statistics using relatively small medical imaging datasets may introduce unstable feature distributions.

Accordingly, BatchNorm statistics are frozen for layers whose parameters remain frozen.

Only BatchNorm layers belonging to trainable backbone blocks continue updating.

This strategy improves optimization stability while preserving pretrained normalization statistics.

---

### Why Test-Time Augmentation?

Retinal lesions should remain detectable under simple geometric transformations.

During inference, predictions from four transformed versions of each image are averaged:

- Original image
- Horizontal Flip
- Vertical Flip
- 90° Rotation

Averaging these predictions reduces prediction variance and generally produces more robust probability estimates, leading to improved Quadratic Weighted Kappa.

---

### Why Quadratic Weighted Kappa?

Unlike Accuracy or Macro F1-score, Quadratic Weighted Kappa incorporates the ordinal distance between prediction errors.

For example,

predicting **Grade 4 as Grade 3** is considered substantially less severe than predicting **Grade 4 as Grade 0**.

Because this behavior closely matches clinical decision making, the entire training pipeline—including the loss function, decision rule, threshold optimization, checkpoint selection, and early stopping—is explicitly optimized with respect to **Quadratic Weighted Kappa**.

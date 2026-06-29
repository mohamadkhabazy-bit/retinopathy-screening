# 🩺 APTOS Diabetic Retinopathy Screening
### EfficientNet-B2 + CBAM + GeM + Two-Phase Fine-Tuning

A deep learning pipeline for automatic grading of **Diabetic Retinopathy (DR)** from retinal fundus images using the **APTOS 2019 Blindness Detection** dataset.

The project follows the **CRISP-DM (Cross Industry Standard Process for Data Mining)** methodology and focuses on building a highly optimized yet lightweight medical image classification system capable of running on consumer-grade hardware.

---

# 🚀 Final Performance

| Metric | Score |
|---------|-------|
| Quadratic Weighted Kappa (QWK) | **0.8806** |
| Accuracy | **82.8%** |
| Macro F1-score | **66.7%** |
| Validation Loss | **0.3865** |

---

# ✨ Features

- EfficientNet-B2 backbone pretrained on ImageNet
- CBAM Attention Module
- GeM Pooling
- Transfer Learning
- Two-Phase Fine-Tuning
- Strict BatchNorm Statistics Locking
- Cosine Annealing Scheduler with Warmup
- Earth Mover's Distance (EMD) Loss
- Cross Entropy with Label Smoothing
- Mixed Precision Training (AMP)
- Gradient Accumulation
- WeightedRandomSampler
- Class Weight Smoothing (Square Root Scaling)
- Anatomy-Aware Data Augmentation
- Ben Graham Preprocessing
- Green Channel Extraction

---

# 💻 Development Environment

The complete project was developed and validated on consumer-grade hardware.

| Component | Specification |
|-----------|---------------|
| GPU | NVIDIA GeForce RTX 4050 Laptop GPU |
| GPU Memory | 6 GB VRAM |
| RAM | 24 GB |
| Framework | PyTorch |
| CUDA | Enabled |

The training pipeline was specifically optimized for limited GPU memory through:

- Mixed Precision (AMP)
- Gradient Accumulation
- Efficient memory management
- Cosine Annealing learning rate scheduling

No high-end workstation or cloud GPU was required.

---

# 📚 CRISP-DM Structure

This repository is organized according to the CRISP-DM methodology.

```
Business Understanding/
Data Understanding/
Data Preparation/
Modeling/
Evaluation/
Deployment/
```

Each directory contains its own documentation (`README.md`) together with the corresponding implementation files.

| Phase | Description |
|--------|-------------|
| Business Understanding | Project objectives, problem definition, success criteria |
| Data Understanding | Dataset exploration and statistical analysis |
| Data Preparation | Image preprocessing, augmentation and sampling |
| Modeling | Model architecture and training pipeline |
| Evaluation | Performance analysis and validation |
| Deployment | Deployment strategy and inference pipeline |

---

# 📂 Repository Structure

```text
.
── Business Understanding/
│   └── README.md
│
├── Data Understanding/
│   └── README.md
│
├── Data Preparation/
│   ├── README.md
│   ├── dataset.py
│   ├── visualize_data.ipynb          
│   └── visualizations/              
│       ├── class_distribution.png
│       ├── sample_images.png
│       ├── image_quality_analysis.png
│       ├── image_dimensions.png
│       ├── clahe_effect.png
│       ├── pipeline_majority.png
│       └── pipeline_minority.png
│
├── Modeling/
│   ├── README.md
│   ├── model.py
│   ├── train_phase1.py
│   └── train_phase2.py
│
├── Evaluation/
│   └── README.md
│
├── Deployment/
│   └── README.md
│
├── checkpoints/
│   └── best_model.pth
│
├── dataset/
│   └── raw data/
│
├── .gitignore
└── README.md
```

---

# 🧠 Model Overview

The proposed model combines several modern deep learning techniques specifically designed for retinal image analysis.

- EfficientNet-B2 feature extractor
- CBAM attention mechanism
- GeM pooling layer
- Fully connected classifier with BatchNorm and Dropout

Training is performed using a carefully designed **Two-Phase Fine-Tuning Strategy** that maximizes transfer learning performance while preventing feature drift through strict BatchNorm statistics locking.

Complete implementation details are available in the **Modeling** directory.

---

# 📊 Evaluation

The final model achieves a **Quadratic Weighted Kappa (QWK) of 0.8806**, demonstrating strong agreement with expert annotations.

Performance was evaluated using:

- Quadratic Weighted Kappa (Primary Metric)
- Accuracy
- Macro Precision
- Macro Recall
- Macro F1-score
- Confusion Matrix

Detailed analysis and discussion are available in the **Evaluation** directory.

---

# 📥 Pretrained Model

The final trained model is included in this repository.

```
checkpoints/
└── best_model.pth
```

Load the model using:

```python
import torch
from model.model import RetinopathyModel

model = RetinopathyModel(num_classes=5)

state_dict = torch.load(
    "checkpoints/best_model.pth",
    map_location="cpu"
)

model.load_state_dict(state_dict)
model.eval()
```

---

# ⚙️ Installation

Clone the repository

```bash
git clone <repository-url>
```

Install dependencies

```bash
pip install torch torchvision timm albumentations datasets scikit-learn opencv-python pillow numpy
```

---

# 🏃 Training

Phase 1

```bash
python train_phase1.py
```

Phase 2

```bash
python train_phase2.py
```

The best-performing model is automatically saved as:

```
checkpoints/best_model.pth
```

---

# 📄 Dataset

This project uses the **APTOS 2019 Blindness Detection** dataset.

Due to licensing restrictions, the dataset is **not included** in this repository.

After downloading the dataset, place it inside:

```
dataset/
└── raw data/
```

---

# 📜 License

This repository is intended for educational and research purposes.

The APTOS 2019 dataset is distributed separately under its original license.

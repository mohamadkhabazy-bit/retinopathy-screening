# 🩺 APTOS Diabetic Retinopathy Screening (EfficientNet-B2 + CBAM + GeM)

A highly optimized, two-phase deep learning pipeline for grading Diabetic Retinopathy (DR) from fundus images. This project achieves a **Quadratic Weighted Kappa (QWK) of 0.8806** on the APTOS 2019 validation set, specifically engineered to run efficiently on consumer GPUs (6GB VRAM).

## 🌟 Key Highlights & Final Performance

* **Final Validation QWK:** `0.8806`
* **Final Validation Accuracy:** `82.8%`
* **Hardware Footprint:** Runs comfortably on **6GB VRAM** (e.g., RTX 4050 Laptop) using Gradient Accumulation and Mixed Precision (AMP).
* **Architecture:** EfficientNet-B2 Backbone + CBAM Attention + GeM Pooling.
* **Training Strategy:** Novel Two-Phase approach with strict BatchNorm statistics locking to prevent feature drift.

---

## 📥 Pre-trained Weights

The final fine-tuned model weights are included directly in this repository.

* **Model File:** `checkpoints/best_model.pth`
* **Performance:** QWK 0.8806 | Accuracy 82.8%

**How to load the weights in your code:**

```python
import torch
from model.model import RetinopathyModel

# 1. Initialize the exact same architecture
model = RetinopathyModel(num_classes=5, dropout=0.5)

# 2. Load the weights from the checkpoints folder
state_dict = torch.load("checkpoints/best_model.pth", map_location="cpu")
model.load_state_dict(state_dict)

# 3. Set to evaluation mode for inference
model.eval()
```

---

## 🏗️ Architecture & Methodology

### 1. Data Preprocessing & Augmentation
Fundus images require specialized preprocessing to preserve fine anatomical details like microaneurysms and hemorrhages.
* **Ben Graham Preprocessing:** Applies local contrast enhancement to remove uneven illumination and highlight lesions.
* **Green Channel Extraction:** Isolates the green channel of the RGB image, where blood vessels and lesions exhibit the highest contrast.
* **Aspect-Ratio Preserving Resize:** Uses `LongestMaxSize` combined with `PadIfNeeded` to scale images while strictly maintaining the true circular geometry of the retina.
* **Anatomy-Aware Augmentations:** Utilizes gentle geometric transforms (flips, 90-degree rotations, slight shifts) to ensure tiny diagnostic lesions are never warped or erased.

### 2. Model Architecture
* **Backbone:** `EfficientNet-B2` (pretrained on ImageNet). Provides an optimal balance of depth and parameter efficiency.
* **CBAM (Convolutional Block Attention Module):** Applied after the backbone to help the network focus on "where" (spatial attention) and "what" (channel attention) the tiny lesions are.
* **GeM (Generalized Mean) Pooling:** Replaces standard Average Pooling. GeM acts as a magnifying glass, emphasizing the highest activations (the lesions) rather than diluting them with background tissue.
* **Classification Head:** A 512-unit fully connected layer with BatchNorm, ReLU, and Dropout (0.5).

### 3. Loss Function
* **Combined Loss:** A weighted combination of **Earth Mover's Distance (EMD)** and **Cross-Entropy (with label smoothing)**. 
* *Why EMD?* DR grades are ordinal (Mild is closer to Moderate than to Proliferative). EMD penalizes the model based on the "distance" between the predicted and true cumulative distributions, respecting the medical severity scale.

### 4. Handling Class Imbalance
The APTOS dataset is heavily skewed towards Class 0 (No DR). To ensure robust learning across all severity levels, we employ a stabilized dual-weighting approach:
* A `WeightedRandomSampler` guarantees balanced batch composition during training.
* The loss function applies **square-root-scaled class weights**. This provides optimal gradient scaling for minority classes while maintaining training stability and preventing over-penalization.

---

## 🧠 The Two-Phase Training Strategy

To maximize the utility of the pretrained backbone while adapting to medical imagery, we utilize a strict Two-Phase training methodology:

### Phase 1: Feature Extraction & Head Training
* **Action:** The entire EfficientNet-B2 backbone is **frozen**. Only the CBAM attention module and the Classification Head are trained.
* **Strict BatchNorm Locking:** We use a custom `freeze_bn_stats()` function. Even when the backbone is frozen, PyTorch's `.train()` mode still updates BatchNorm running means/variances. We force frozen BN layers into `.eval()` mode every epoch to lock them to their pretrained ImageNet statistics, ensuring perfectly stable feature extraction.
* **Scheduler:** Cosine Annealing with a 3-epoch warmup.

### Phase 2: Surgical Fine-Tuning
* **Action:** We unfreeze **ONLY the very last block** of the EfficientNet-B2 backbone to allow deep semantic features to adapt to retinal pathology.
* **Discriminative LRs:** The backbone uses a microscopic learning rate (`3e-5`), while the Head uses a higher rate (`1e-4`). 
* **Warmup:** A 5-epoch linear warmup is applied to allow the newly unfrozen BatchNorm layers to gently stabilize their statistics before the convolution weights start making significant updates.

---

## 📂 Project Structure

```text
├── dataset/
│   ├── raw data/             # Local parquet files of the APTOS dataset
│   └── dataset.py            # Data loading, Ben Graham, Augmentations, Samplers
├── checkpoints/              
│   └── best_model.pth        # 🏆 The final champion model (QWK 0.8806)
├── model/
│   └── model.py              # EfficientNet, CBAM, GeM, Loss, Training Loops
├── train_phase1.py           # Phase 1 execution script (Frozen backbone)
├── train_phase2.py           # Phase 2 execution script (Unfreeze last block)
├── .gitignore                # Git ignore rules
└── README.md
```

---

## 🚀 Setup & Installation

### 1. Prerequisites
* Python 3.9+
* CUDA-enabled GPU (Tested on 6GB VRAM)

### 2. Install Dependencies

```bash
pip install torch torchvision timm albumentations datasets scikit-learn opencv-python pillow numpy
```

### 3. Configure Cache Directories
This project forces HuggingFace and PyTorch caches to a specific drive to prevent C: drive overflow. **Update the environment variables at the top of the python scripts to match your local paths:**

```python
os.environ["HF_HOME"]           = r"YOUR_PATH\hf_home"
os.environ["HF_DATASETS_CACHE"] = r"YOUR_PATH\hf_cache"
os.environ["TORCH_HOME"]        = r"YOUR_PATH\torch_cache"
```

---

## 🏃 How to Train from Scratch

### Step 1: Run Phase 1
Phase 1 trains the attention mechanism and the classification head while keeping the feature extractor stable.

```bash
python train_phase1.py
```

*Wait for Phase 1 to complete. It will save the best weights to `checkpoints/best_model.pth`.*

### Step 2: Run Phase 2
Phase 2 surgically fine-tunes the deepest layers of the backbone.

```bash
python train_phase2.py
```

*Note: Phase 2 will automatically load the Phase 1 weights, evaluate them to set a baseline, and will ONLY overwrite `best_model.pth` if Phase 2 achieves a higher QWK.*

---

## 📜 License

This project is created for educational and research purposes.
*Note: The APTOS dataset is provided by EyePacs under the CC BY-NC 4.0 license.*

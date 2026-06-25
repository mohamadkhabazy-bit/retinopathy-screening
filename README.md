# 🩺 APTOS Diabetic Retinopathy Screening (EfficientNet-B2 + CBAM + GeM)

A highly optimized, two-phase deep learning pipeline for grading Diabetic Retinopathy (DR) from fundus images. This project achieves a **Quadratic Weighted Kappa (QWK) of 0.8787** on the APTOS 2019 validation set, specifically engineered to run efficiently on consumer GPUs (6GB VRAM).

## 🌟 Key Highlights & Final Performance

* **Final Validation QWK:** `0.8787`
* **Final Validation Accuracy:** `~82.5%`
* **Hardware Footprint:** Runs comfortably on **6GB VRAM** (e.g., RTX 4050 Laptop) using Gradient Accumulation and Mixed Precision (AMP).
* **Architecture:** EfficientNet-B2 Backbone + CBAM Attention + GeM Pooling.
* **Training Strategy:** Novel Two-Phase approach with strict BatchNorm statistics locking to prevent catastrophic forgetting.

---

## 🏗️ Architecture & Methodology

### 1. Data Preprocessing & Augmentation
Medical imaging requires specialized preprocessing. Standard ImageNet augmentations often destroy the tiny microaneurysms required for DR grading.
* **Ben Graham Preprocessing:** Applies local contrast enhancement to remove uneven illumination and highlight lesions.
* **Green Channel Extraction:** Isolates the green channel of the RGB image, where blood vessels and hemorrhages have the highest contrast.
* **Aspect-Ratio Preserving Resize:** Uses `LongestMaxSize` + `PadIfNeeded` instead of standard `Resize`. This prevents the stretching/squishing of retinal anatomy.
* **Gentle Augmentations:** We explicitly **removed** `ElasticTransform`, `GridDistortion`, and `CoarseDropout`. These transforms warp the retina or drop out pixels, which can literally erase the tiny lesions needed to distinguish between Grade 1 and Grade 2.

### 2. Model Architecture
* **Backbone:** `EfficientNet-B2` (pretrained on ImageNet). Provides a great balance of depth and parameter efficiency.
* **CBAM (Convolutional Block Attention Module):** Applied after the backbone to help the network focus on "where" (spatial attention) and "what" (channel attention) the tiny lesions are.
* **GeM (Generalized Mean) Pooling:** Replaces standard Average Pooling. GeM acts as a magnifying glass, emphasizing the highest activations (the lesions) rather than diluting them with background tissue.
* **Classification Head:** A 512-unit fully connected layer with BatchNorm, ReLU, and Dropout (0.5).

### 3. Loss Function
* **Combined Loss:** A weighted combination of **Earth Mover's Distance (EMD)** and **Cross-Entropy (with label smoothing)**. 
* *Why EMD?* DR grades are ordinal (Mild is closer to Moderate than to Proliferative). EMD penalizes the model based on the "distance" between the predicted and true cumulative distributions, respecting the medical severity scale.

### 4. Handling Class Imbalance (The "Softened" Approach)
The APTOS dataset is heavily skewed towards Class 0 (No DR). 
* We use a `WeightedRandomSampler` to oversample minority classes during batching.
* **Crucial Fix:** We apply the **square root (`np.sqrt`)** to the computed class weights before passing them to the Loss function. Applying full class weights *alongside* the sampler causes a "double penalty" that leads to violent overfitting on minority classes. Softening the weights provides the perfect balance.

---

## 🧠 The Two-Phase Training Strategy

Fine-tuning medical images is notoriously difficult due to **BatchNorm Shock** (when unfreezing a backbone causes its running statistics to drift, destroying the learned features). We solve this with a strict Two-Phase approach:

### Phase 1: Feature Extraction & Head Training
* **Action:** The entire EfficientNet-B2 backbone is **frozen**. Only CBAM and the Head are trained.
* **BatchNorm Lock:** We use a custom `freeze_bn_stats()` function. Even though the backbone is frozen, PyTorch's `.train()` mode still updates BatchNorm running means/variances. We force frozen BN layers into `.eval()` mode every epoch to lock them to their pretrained ImageNet statistics.
* **Scheduler:** Cosine Annealing with a 3-epoch warmup.

### Phase 2: Surgical Fine-Tuning
* **Action:** We unfreeze **ONLY the very last block** of the EfficientNet-B2 backbone. 
* **Discriminative LRs:** The backbone uses a microscopic learning rate (`3e-5`), while the Head uses a higher rate (`1e-4`). 
* **Warmup:** A 5-epoch linear warmup is applied to allow the newly unfrozen BatchNorm layers to gently stabilize their statistics before the convolution weights start making significant updates.

---

## 📂 Project Structure

```text
├── dataset/
│   └── raw data/             # Local parquet files of the APTOS dataset
├── checkpoints/              # Saved model weights
│   ├── best_model.pth        # The absolute best model across both phases
│   ├── resume_p1.pth         # Phase 1 training state
│   └── resume_p2.pth         # Phase 2 training state
├── dataset/
│   └── dataset.py            # Data loading, Ben Graham, Augmentations, Samplers
├── model/
│   └── model.py              # EfficientNet, CBAM, GeM, Loss, Training Loops
├── train_phase1.py           # Phase 1 execution script (Frozen backbone)
├── train_phase2.py           # Phase 2 execution script (Unfreeze last block)
└── README.md

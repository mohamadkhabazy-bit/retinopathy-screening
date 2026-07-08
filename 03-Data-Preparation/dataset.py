# %%
import os
import sys

ROOT_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

import cv2
import numpy as np

import torch
from torch.utils.data import Dataset, WeightedRandomSampler

import albumentations as A
from albumentations.pytorch import ToTensorV2

from datasets import load_dataset
from sklearn.utils.class_weight import compute_class_weight

from sklearn.model_selection import train_test_split
from datasets import DatasetDict

from config import (
    RAW_DATA_DIR,
    CACHE_DIR,
    HF_HOME,
    HF_DATASETS_CACHE,
    HF_HUB_CACHE,
    HF_DATASET_NAME
)

os.environ["HF_HOME"] = HF_HOME
os.environ["HF_DATASETS_CACHE"] = HF_DATASETS_CACHE
os.environ["HF_HUB_CACHE"] = HF_HUB_CACHE


# ──────────────────────────────────────────────────────────────
# Constants 
# ──────────────────────────────────────────────────────────────
IMAGE_SIZE       = 512
MINORITY_CLASSES = {1, 3, 4}
MEAN             = [0.485, 0.456, 0.406]
STD              = [0.229, 0.224, 0.225]


# ──────────────────────────────────────────────────────────────
# Load 
# ──────────────────────────────────────────────────────────────
def load_aptos_dataset(
    raw_data_dir: str = RAW_DATA_DIR,
    cache_dir: str = CACHE_DIR
):

    print(f"Loading APTOS dataset from local files: {raw_data_dir}")
    print(f"Arrow cache will be written to: {cache_dir}")

    if raw_data_dir is not None:
        print(f"Loading local parquet dataset from: {raw_data_dir}")
        ds = load_dataset(
            "parquet",
            data_dir=raw_data_dir,
            cache_dir=cache_dir
        )["train"]
    else:
        print(f"Loading HuggingFace dataset: {HF_DATASET_NAME}")
        dataset_dict = load_dataset(
            HF_DATASET_NAME,
            cache_dir=cache_dir
        )
        print(dataset_dict)
        ds = dataset_dict["train"]

    # ==============================
    # Fix String Labels (Crucial for Colab / HF Hub)
    # ==============================
    # HF Hub datasets sometimes return string labels instead of integers.
    # We map them to standard APTOS integers (0-4) to prevent sklearn errors.
    if len(ds) > 0 and isinstance(ds[0]["label"], str):
        print("Detected string labels. Mapping to integers (0-4)...")
        label_map = {
            "no_diabetic_retinopathy": 0,
            "mild_retinopathy": 1,
            "moderate_retinopathy": 2,
            "severe_retinopathy": 3,
            "proliferative_retinopathy": 4
        }
        
        def encode_labels(examples):
            examples["label"] = [label_map[l] for l in examples["label"]]
            return examples
            
        ds = ds.map(encode_labels, batched=True)

    # ==============================
    # Stratified Split
    # ==============================
    indices = np.arange(len(ds))
    labels = np.array(ds["label"])

    train_idx, val_idx = train_test_split(
        indices,
        test_size=0.2,
        stratify=labels,
        random_state=42
    )

    dataset = DatasetDict({
        "train": ds.select(train_idx),
        "test": ds.select(val_idx)
    })

    print(
        f"Total: {len(ds)} | "
        f"Train: {len(dataset['train'])} | "
        f"Val: {len(dataset['test'])}"
    )

    print("\nTrain class distribution:")
    train_labels = dataset["train"]["label"]
    unique, counts = np.unique(
        train_labels,
        return_counts=True
    )
    for cls, count in zip(unique, counts):
        print(f"  Class {cls}: {count}")

    return (
        dataset["train"],
        dataset["test"]
    )


# ──────────────────────────────────────────────────────────────
# Ben Graham Preprocessing
# ──────────────────────────────────────────────────────────────
def ben_graham_preprocess(img: np.ndarray, sigma: int = 10) -> np.ndarray:
    blurred   = cv2.GaussianBlur(img, (0, 0), sigma)
    processed = cv2.addWeighted(img, 4, blurred, -4, 128)
    return processed


# ──────────────────────────────────────────────────────────────
# Green Channel Extraction
# ──────────────────────────────────────────────────────────────
def extract_green_channel(img: np.ndarray) -> np.ndarray:
    green = img[:, :, 1]
    return np.stack([green, green, green], axis=2)


# ──────────────────────────────────────────────────────────────
# Augmentations
# ──────────────────────────────────────────────────────────────
def get_transforms(split: str, image_size: int = IMAGE_SIZE) -> A.Compose:
    assert split in ("val", "test", "majority", "minority"), \
        f"Invalid split: {split}"

    if split in ("val", "test"):
        return A.Compose([
            A.LongestMaxSize(max_size=image_size),
            A.PadIfNeeded(min_height=image_size, min_width=image_size,
                          border_mode=cv2.BORDER_CONSTANT, fill=0),
            A.Normalize(mean=MEAN, std=STD),
            ToTensorV2()
        ])

    if split == "majority":
        return A.Compose([
            A.LongestMaxSize(max_size=image_size),
            A.PadIfNeeded(min_height=image_size, min_width=image_size,
                          border_mode=cv2.BORDER_CONSTANT, fill=0),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.RandomBrightnessContrast(p=0.3),
            A.Normalize(mean=MEAN, std=STD),
            ToTensorV2()
        ])

    if split == "minority":
        return A.Compose([
            A.LongestMaxSize(max_size=image_size),
            A.PadIfNeeded(min_height=image_size, min_width=image_size,
                          border_mode=cv2.BORDER_CONSTANT, fill=0),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.RandomBrightnessContrast(p=0.4),
            A.Affine(
                scale=(0.95, 1.05),
                translate_percent=(0.05, 0.05),
                rotate=(-15, 15),
                p=0.5
            ),
            A.Affine(shear=(-10, 10), p=0.3),
            A.CLAHE(clip_limit=2.0, p=0.3),
            A.Normalize(mean=MEAN, std=STD),
            ToTensorV2()
        ])


# ──────────────────────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────────────────────
class APTOSDataset(Dataset):

    def __init__(
        self,
        hf_dataset,
        split,
        image_size=IMAGE_SIZE,
        use_ben_graham=True,
        use_green_channel=True
    ):
        assert split in ("train", "val", "test"), \
            f"Invalid split: '{split}'"

        self.ds                = hf_dataset
        self.split             = split
        self.use_ben_graham    = use_ben_graham
        self.use_green_channel = use_green_channel

        self.majority_tf = get_transforms("majority", image_size)
        self.minority_tf = get_transforms("minority", image_size)
        self.val_tf      = get_transforms("val", image_size)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        sample = self.ds[idx]
        label  = int(sample["label"])

        img = sample["image"].convert("RGB")
        img = np.array(img)

        if self.use_ben_graham:
            img = ben_graham_preprocess(img)

        if self.use_green_channel:
            img = extract_green_channel(img)

        if self.split == "train":
            tf = self.minority_tf if label in MINORITY_CLASSES \
                 else self.majority_tf
        else:
            tf = self.val_tf

        augmented = tf(image=img)
        return (
            augmented["image"],
            torch.tensor(label, dtype=torch.long)
        )


# ──────────────────────────────────────────────────────────────
# Class Weights + Sampler
# ──────────────────────────────────────────────────────────────
def get_class_weights(labels) -> torch.Tensor:
    weights = compute_class_weight(
        class_weight="balanced",
        classes=np.array([0, 1, 2, 3, 4]),
        y=labels
    )
    weights = np.sqrt(weights)
    print("Class weights:")
    for i, w in enumerate(weights):
        print(f"  Class {i}: {w:.3f}")
    return torch.FloatTensor(weights)


def get_sampler(hf_dataset) -> WeightedRandomSampler:
    labels         = hf_dataset["label"]
    class_weights  = get_class_weights(labels)
    sample_weights = [float(class_weights[l]) for l in labels]

    return WeightedRandomSampler(
        weights     = torch.FloatTensor(sample_weights),
        num_samples = len(sample_weights),
        replacement = True
    )


# ──────────────────────────────────────────────────────────────
# Sanity Check
# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from torch.utils.data import DataLoader

    train_ds, val_ds = load_aptos_dataset()

    train_dataset = APTOSDataset(train_ds, split="train")
    val_dataset   = APTOSDataset(val_ds,   split="val")

    sampler = get_sampler(train_ds)

    train_loader = DataLoader(
        train_dataset, batch_size=4,
        sampler=sampler, num_workers=0, pin_memory=True
    )

    images, labels = next(iter(train_loader))
    print(f"\n✅ Batch loaded successfully")
    print(f"   Image shape : {images.shape}")
    print(f"   Labels      : {labels.tolist()}")

# %%
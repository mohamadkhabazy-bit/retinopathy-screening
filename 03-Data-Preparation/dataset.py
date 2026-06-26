# %%
import os
os.environ["HF_HOME"]            = r"E:\retinopathy-screening\hf_home"
os.environ["HF_DATASETS_CACHE"]  = r"E:\retinopathy-screening\hf_cache"
os.environ["HF_HUB_CACHE"]       = r"E:\retinopathy-screening\hf_hub_cache"

import io
import cv2
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset, WeightedRandomSampler

import albumentations as A
from albumentations.pytorch import ToTensorV2

from datasets import load_dataset
from sklearn.utils.class_weight import compute_class_weight


# ──────────────────────────────────────────────────────────────
# Constants — everything on E:
# ──────────────────────────────────────────────────────────────
IMAGE_SIZE       = 512
MINORITY_CLASSES = {1, 3, 4}
MEAN             = [0.485, 0.456, 0.406]
STD              = [0.229, 0.224, 0.225]

RAW_DATA_DIR = r"E:\retinopathy-screening\dataset\raw data"
CACHE_DIR    = r"E:\retinopathy-screening\hf_cache"


# ──────────────────────────────────────────────────────────────
# Load — reads local parquet files, writes Arrow cache to E: too
# ──────────────────────────────────────────────────────────────
def load_aptos_dataset(
    raw_data_dir: str = RAW_DATA_DIR,
    cache_dir: str = CACHE_DIR
):
    """
    Loads the 13 local parquet files AND forces the converted Arrow
    cache to also live on E: — without explicitly passing cache_dir
    here, load_dataset falls back to the C: default even though the
    HF_DATASETS_CACHE env var is set above, since some datasets
    versions only respect the env var for the Hub cache, not for
    builder-level caching. Passing it explicitly closes that gap.
    """
    print(f"Loading APTOS dataset from local files: {raw_data_dir}")
    print(f"Arrow cache will be written to: {cache_dir}")

    ds = load_dataset(
        "parquet",
        data_dir=raw_data_dir,
        cache_dir=cache_dir
    )["train"]

    ds = ds.train_test_split(
        test_size=0.2,
        stratify_by_column="label",
        seed=42
    )

    print(f"Train: {len(ds['train'])} | Val: {len(ds['test'])}")
    print("\nTrain class distribution:")
    train_labels = ds["train"]["label"]
    unique, counts = np.unique(train_labels, return_counts=True)
    for cls, count in zip(unique, counts):
        print(f"  Class {cls}: {count}")

    return ds["train"], ds["test"]


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
            A.ShiftScaleRotate(
                shift_limit=0.05,
                scale_limit=0.05,
                rotate_limit=15,
                p=0.5
            ),
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

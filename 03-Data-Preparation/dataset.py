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

    # ✅ اصلاح نام‌گذاری: تغییر "test" به "val" برای هماهنگی با کل پروژه
    dataset = DatasetDict({
        "train": ds.select(train_idx),
        "val": ds.select(val_idx)
    })

    print(
        f"Total: {len(ds)} | "
        f"Train: {len(dataset['train'])} | "
        f"Val: {len(dataset['val'])}"
    )

    print("\nTrain class distribution:")
    train_labels = dataset["train"]["label"]
    unique, counts = np.unique(
        train_labels,
        return_counts=True
    )
    for cls, count in zip(unique, counts):
        print(f"  Class {cls}: {count}")

    # ✅ ریترن کردن با کلید صحیح "val"
    return (
        dataset["train"],
        dataset["val"]
    )


# ──────────────────────────────────────────────────────────────
# Crop Black Borders
# ──────────────────────────────────────────────────────────────
def crop_image_from_gray(img: np.ndarray, tol: int = 7) -> np.ndarray:
    """
    برش خودکار حاشیه‌های سیاه تصویر شبکیه با حفظ کامل ساختار دایره.
    tol: آستانه سیاه بودن پیکسل‌ها (بین 0 تا 255)
    """
    if img.ndim == 2:
        mask = img > tol
        return img[np.ix_(mask.any(axis=1), mask.any(axis=0))]
    
    elif img.ndim == 3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        mask = gray_img > tol
        
        check_shape = img[np.ix_(mask.any(axis=1), mask.any(axis=0))].shape[0]
        if check_shape == 0:
            return img
        else:
            img1 = img[np.ix_(mask.any(axis=1), mask.any(axis=0))]
            return img1
    
    return img



# ──────────────────────────────────────────────────────────────
# CLAHE on LAB Color Space Preprocessing
# ──────────────────────────────────────────────────────────────
def clahe_lab_preprocess(img: np.ndarray, clip_limit: float = 2.0, tile_grid_size: tuple = (8, 8)) -> np.ndarray:
    """
    اعمال الگوریتم CLAHE روی کانال روشنایی (L) در فضای رنگی LAB.
    """
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    cl = clahe.apply(l_channel)
    
    merged = cv2.merge((cl, a_channel, b_channel))
    img_rgb = cv2.cvtColor(merged, cv2.COLOR_LAB2RGB)
    
    return img_rgb

# ──────────────────────────────────────────────────────────────
# Augmentations (Safe for Medical Images)
# ─────────────────────────────────────────────────────────────
def get_transforms(split: str, image_size: int = IMAGE_SIZE) -> A.Compose:
    assert split in ("train", "val", "test"), \
        f"Invalid split: {split}"

    if split in ("val", "test"):
        return A.Compose([
            A.LongestMaxSize(max_size=image_size),
            A.PadIfNeeded(min_height=image_size, min_width=image_size,
                          border_mode=cv2.BORDER_CONSTANT, fill=0),
            A.Normalize(mean=MEAN, std=STD),
            ToTensorV2()
        ])

    # پایپ‌لاین واحد و امن برای آموزش
    if split == "train":
        return A.Compose([
            A.LongestMaxSize(max_size=image_size),
            A.PadIfNeeded(min_height=image_size, min_width=image_size,
                          border_mode=cv2.BORDER_CONSTANT, fill=0),
            
            # ۱. تغییرات هندسی (بدون از دست رفتن اطلاعات پیکسل)
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            
            A.ShiftScaleRotate(
                shift_limit=0.05, 
                scale_limit=0.05, 
                rotate_limit=15, 
                border_mode=cv2.BORDER_CONSTANT, 
                fill=0, 
                p=0.5
            ),
            
            # ۲. تغییرات نوری کنترل‌شده (بسیار ملایم‌تر از قبل برای حفظ اثر CLAHE)
            A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.3),
            
            # ۳. بلر بسیار محدود (فقط برای ایجاد پایداری در تصاویر خارج از فوکوس)
            A.OneOf([
                A.GaussianBlur(blur_limit=(3, 5), p=0.5),
                A.MotionBlur(blur_limit=3, p=0.5),
            ], p=0.1),  # احتمال کل را به 10% کاهش دادیم
            
            # ۴. نویز بسیار ضعیف
            A.GaussNoise(var_limit=(10.0, 20.0), p=0.1),  # واریانس و احتمال کم شد
            
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
        use_clahe_lab=True
    ):
        assert split in ("train", "val", "test"), \
            f"Invalid split: '{split}'"

        self.ds            = hf_dataset
        self.split         = split
        self.use_clahe_lab = use_clahe_lab

        self.train_tf = get_transforms("train", image_size)
        self.val_tf   = get_transforms("val", image_size)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        sample = self.ds[idx]
        label  = int(sample["label"])

        # ۱. لود تصویر
        img = sample["image"].convert("RGB")
        img = np.array(img)

        # ۲. حذف حاشیه‌های سیاه
        img = crop_image_from_gray(img)

        # ۳. اعمال پیش‌پردازش LAB
        if self.use_clahe_lab:
            img = clahe_lab_preprocess(img)

        # ۴. انتخاب ترنسفورم
        if self.split == "train":
            tf = self.train_tf
        else:
            tf = self.val_tf

        # ۵. اعمال نهایی و تبدیل به تنسور
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
    # ✅ حذف np.sqrt برای بالانس شدن کامل و ۱۰٪ فیزیکی بچ‌ها توسط سامپلر
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
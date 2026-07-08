import os
import platform


# ==============================
# Detect Environment
# ==============================

IS_COLAB = "COLAB_GPU" in os.environ
IS_WINDOWS = platform.system() == "Windows"


# ==============================
# Project Root
# ==============================

if IS_WINDOWS:

    PROJECT_ROOT = r"E:\retinopathy-screening"

elif IS_COLAB:

    PROJECT_ROOT = "/content/retinopathy-screening"

else:

    PROJECT_ROOT = os.getcwd()


# ==============================
# Dataset Paths
# ==============================

if IS_WINDOWS:

    # Local parquet files
    RAW_DATA_DIR = os.path.join(
        PROJECT_ROOT,
        "dataset",
        "raw data"
    )

else:

    # Colab will load dataset from HuggingFace
    RAW_DATA_DIR = None


# HuggingFace dataset name (used when RAW_DATA_DIR is None)
HF_DATASET_NAME = "sngsfydy/aptos_train"

# ==============================
# HuggingFace / Torch Cache
# ==============================

CACHE_DIR = os.path.join(
    PROJECT_ROOT,
    "hf_cache"
)

HF_HOME = os.path.join(
    PROJECT_ROOT,
    "hf_home"
)



HF_DATASETS_CACHE = os.path.join(
    PROJECT_ROOT,
    "hf_cache"
)

HF_HUB_CACHE = os.path.join(
    PROJECT_ROOT,
    "hf_hub_cache"
)

TORCH_HOME = os.path.join(
    PROJECT_ROOT,
    "torch_cache"
)


# ==============================
# Checkpoints
# ==============================

CHECKPOINT_DIR = os.path.join(
    PROJECT_ROOT,
    "checkpoints"
)

BEST_MODEL_PATH = os.path.join(
    CHECKPOINT_DIR,
    "best_model.pth"
)

RESUME_P1_PATH = os.path.join(
    CHECKPOINT_DIR,
    "resume_p1.pth"
)

RESUME_P2_PATH = os.path.join(
    CHECKPOINT_DIR,
    "resume_p2.pth"
)


# ==============================
# Create required folders
# ==============================

os.makedirs(CHECKPOINT_DIR, exist_ok=True)



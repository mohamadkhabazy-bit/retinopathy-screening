import os
import sys

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
for folder in os.listdir(ROOT_DIR):
    folder_path = os.path.join(ROOT_DIR, folder)
    if os.path.isdir(folder_path):
        sys.path.append(folder_path)

from config import (
    HF_HOME, HF_DATASETS_CACHE, HF_HUB_CACHE, TORCH_HOME,
    CHECKPOINT_DIR, BEST_MODEL_PATH, BEST_MODEL_P2_PATH,
    RESUME_P1_PATH, RESUME_P2_PATH,
)

os.environ["HF_HOME"] = HF_HOME
os.environ["HF_DATASETS_CACHE"] = HF_DATASETS_CACHE
os.environ["HF_HUB_CACHE"] = HF_HUB_CACHE
os.environ["TORCH_HOME"] = TORCH_HOME
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import numpy as np
from torch.utils.data import DataLoader

from dataset import load_aptos_dataset, APTOSDataset, get_sampler
from model import (
    RetinopathyModel, set_seed, unfreeze_last_blocks,
    get_loss_fn, train, load_full_checkpoint, load_best_model,
    final_evaluation, model_summary, validate
)

# ──────────────────────────────────────────────────────────────
# Hyperparameters - Phase 2 (With Frozen BN)
# ──────────────────────────────────────────────────────────────
BATCH_SIZE          = 32
ACCUMULATION_STEPS  = 2   
EPOCHS              = 25      
ES_PATIENCE         = 15      

NUM_WORKERS_TRAIN   = 2
NUM_WORKERS_VAL     = 1

# ✅ 1 بلاک آخر آن‌فریز می‌شود (با BNهای قفل‌شده)
NUM_BLOCKS_TO_UNFREEZE = 1    

def build_optimizer(model: torch.nn.Module) -> torch.optim.Optimizer:
    backbone_params = [p for n, p in model.named_parameters() if p.requires_grad and n.startswith("backbone")]
    head_params     = [p for n, p in model.named_parameters() if p.requires_grad and not n.startswith("backbone")]
    
    print(f"  Backbone trainable params: {sum(p.numel() for p in backbone_params):,}")
    print(f"  Head/CBAM params: {sum(p.numel() for p in head_params):,}")
    
    return torch.optim.AdamW([
        # ✅ LR مناسب برای backbone با BNهای قفل‌شده
        {"params": backbone_params, "lr": 5e-5},  
        # ✅ LR بالاتر برای هد
        {"params": head_params,     "lr": 1e-4},
    ], weight_decay=0.0001)

def main():
    set_seed(42)
    torch.cuda.empty_cache()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

    train_ds, val_ds = load_aptos_dataset()
    train_dataset = APTOSDataset(train_ds, split="train")
    val_dataset   = APTOSDataset(val_ds,   split="val")

    sampler = get_sampler(train_ds)

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, sampler=sampler,
        num_workers=NUM_WORKERS_TRAIN, pin_memory=True,
        persistent_workers=True, prefetch_factor=2, drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS_VAL, pin_memory=True,
        persistent_workers=True, drop_last=False
    )
    
    print(f"Effective batch size: {BATCH_SIZE * ACCUMULATION_STEPS}")

    model = RetinopathyModel(num_classes=5, dropout=0.4).to(device)
    loss_fn = get_loss_fn(class_weights=None, alpha=0.7, label_smoothing=0.05).to(device)

    start_epoch      = 1
    initial_best_qwk = -float("inf")
    scheduler_state  = None

    if os.path.exists(RESUME_P2_PATH):
        print("\nFound Phase 2 checkpoint — resuming.")
        peek_ckpt = torch.load(RESUME_P2_PATH, map_location=device, weights_only=False)
        num_blocks = peek_ckpt.get("num_blocks", NUM_BLOCKS_TO_UNFREEZE)
        
        unfreeze_last_blocks(model, num_blocks=num_blocks, unfreeze_conv_head=False)
        optimizer = build_optimizer(model)

        loaded_epoch, best_qwk, sched_state, _ = load_full_checkpoint(
            model, optimizer, RESUME_P2_PATH, device
        )
        start_epoch      = loaded_epoch + 1
        initial_best_qwk = best_qwk
        scheduler_state  = sched_state
    else:
        print("\nNo Phase 2 checkpoint — loading best Phase 1 weights to start fresh.")
        model = load_best_model(model, BEST_MODEL_PATH, device)

        print("Evaluating loaded Phase 1 model to set the Phase 2 baseline...")
        _, p1_metrics = validate(model, val_loader, loss_fn, device, print_report=False)
        initial_best_qwk = p1_metrics["qwk"]
        print(f"  Phase 1 baseline QWK (must beat this to save): {initial_best_qwk:.4f}")

        num_blocks = NUM_BLOCKS_TO_UNFREEZE
        # ✅ آن‌فریز کردن 1 بلاک آخر (با BNهای قفل‌شده توسط freeze_bn_stats)
        unfreeze_last_blocks(model, num_blocks=num_blocks, unfreeze_conv_head=False)
        optimizer = build_optimizer(model)

    model_summary(model)

    if start_epoch > EPOCHS:
        print(f"Phase 2 already completed. Skipping to final evaluation.")
    else:
        history = train(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device=device,
            num_epochs=EPOCHS,
            es_patience=ES_PATIENCE,
            start_epoch=start_epoch,
            initial_best_qwk=initial_best_qwk,
            scheduler_state=scheduler_state,
            checkpoint_path=BEST_MODEL_P2_PATH,
            resume_path=RESUME_P2_PATH,
            accumulation_steps=ACCUMULATION_STEPS,
            checkpoint_extra={"num_blocks": num_blocks},
            scheduler_type="cosine",
            warmup_epochs=5,
        )

    # ارزیابی نهایی با TTA و ذخیره آستانه‌ها
    model = load_best_model(model, BEST_MODEL_P2_PATH, device)
    metrics, thresholds = final_evaluation(
        model, val_loader, loss_fn, device, 
        tta=True, 
        tune_thresholds=True
    )
    
    thresholds_path = os.path.join(CHECKPOINT_DIR, "val_thresholds.npy")
    np.save(thresholds_path, thresholds)
    print(f"\n✅ Thresholds saved to: {thresholds_path}")

if __name__ == "__main__":
    main()
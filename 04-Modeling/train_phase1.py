import os
os.environ["HF_HOME"]              = r"E:\retinopathy-screening\hf_home"
os.environ["HF_DATASETS_CACHE"]    = r"E:\retinopathy-screening\hf_cache"
os.environ["HF_HUB_CACHE"]         = r"E:\retinopathy-screening\hf_hub_cache"
os.environ["TORCH_HOME"]           = r"E:\retinopathy-screening\torch_cache"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
from torch.utils.data import DataLoader

from dataset.dataset import load_aptos_dataset, APTOSDataset, get_sampler, get_class_weights
from model.model import (
    RetinopathyModel, set_seed, freeze_backbone, get_loss_fn,
    train, load_full_checkpoint, model_summary
)

ROOT            = r"E:\retinopathy-screening"
CHECKPOINT_DIR  = os.path.join(ROOT, "checkpoints")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

BEST_MODEL_PATH = os.path.join(CHECKPOINT_DIR, "best_model.pth")
RESUME_PATH     = os.path.join(CHECKPOINT_DIR, "resume_p1.pth")

BATCH_SIZE          = 8
ACCUMULATION_STEPS  = 2

EPOCHS              = 30

ES_PATIENCE         = 10

LEARNING_RATE       = 3e-4

WEIGHT_DECAY        = 0.01

DROPOUT             = 0.5

SCHEDULER_TYPE      = "cosine"
WARMUP_EPOCHS       = 3

NUM_WORKERS_TRAIN   = 4
NUM_WORKERS_VAL     = 2


def main():
    set_seed(42)
    torch.cuda.empty_cache()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU : {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    train_ds, val_ds = load_aptos_dataset()

    train_dataset = APTOSDataset(train_ds, split="train")
    val_dataset   = APTOSDataset(val_ds,   split="val")

    sampler = get_sampler(train_ds)

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE,
        sampler=sampler,
        num_workers=NUM_WORKERS_TRAIN,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
        drop_last=True,   # avoids a leftover batch of size 1 hitting BatchNorm
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS_VAL,
        pin_memory=True,
        persistent_workers=True,
        drop_last=False,  # keep every validation image for accurate metrics
    )
    print(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")
    print(f"Effective batch size: {BATCH_SIZE * ACCUMULATION_STEPS}")
    print(f"Train workers: {NUM_WORKERS_TRAIN} | Val workers: {NUM_WORKERS_VAL}")

    model = RetinopathyModel(num_classes=5, dropout=DROPOUT).to(device)
    class_weights = get_class_weights(train_ds["label"]).to(device)
    loss_fn = get_loss_fn(class_weights, alpha=0.7).to(device)


    freeze_backbone(model)
    model_summary(model)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )

    start_epoch      = 1
    initial_best_qwk = -float("inf")
    scheduler_state   = None

    if os.path.exists(RESUME_PATH):
        print(f"\nFound existing checkpoint — resuming Phase 1.")
        loaded_epoch, best_qwk, sched_state, _ = load_full_checkpoint(
            model, optimizer, RESUME_PATH, device
        )
        start_epoch      = loaded_epoch + 1
        initial_best_qwk = best_qwk
        scheduler_state   = sched_state
    else:
        print("\nNo checkpoint found — starting Phase 1 fresh.")

    if start_epoch > EPOCHS:
        print(f"Phase 1 already completed ({EPOCHS} epochs).")
        return

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
        checkpoint_path=BEST_MODEL_PATH,
        resume_path=RESUME_PATH,
        accumulation_steps=ACCUMULATION_STEPS,
        scheduler_type=SCHEDULER_TYPE,
        warmup_epochs=WARMUP_EPOCHS,
    )

    print("\nPhase 1 complete.")


if __name__ == "__main__":
    main()

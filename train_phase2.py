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
    RetinopathyModel, set_seed, unfreeze_last_blocks, set_finetune_lr,
    get_loss_fn, train, load_full_checkpoint, load_best_model,
    final_evaluation, model_summary, validate
)

ROOT            = r"E:\retinopathy-screening"
CHECKPOINT_DIR  = os.path.join(ROOT, "checkpoints")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

BEST_MODEL_PATH = os.path.join(CHECKPOINT_DIR, "best_model.pth")
RESUME_PATH     = os.path.join(CHECKPOINT_DIR, "resume_p2.pth")

BATCH_SIZE          = 8   # was 4 — real batches of 8 give BatchNorm and
                           # gradients a less noisy signal than 4+accumulation
ACCUMULATION_STEPS  = 1   # was 2 — no longer needed; effective batch stays
                           # at 8 (same as before: 4*2), just computed in one
                           # real pass instead of two accumulated halves
EPOCHS              = 50
ES_PATIENCE         = 10   # was 7 — fine-tuning signal is noisier/slower
                           # than Phase 1; give the LR scheduler's drops
                           # more time to take effect before stopping.

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
        drop_last=True,   # ✅ avoids a leftover batch of size 1 hitting BatchNorm
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

    model = RetinopathyModel(num_classes=5, dropout=0.4).to(device)
    class_weights = get_class_weights(train_ds["label"]).to(device)
    loss_fn = get_loss_fn(class_weights, alpha=0.7).to(device)

    start_epoch      = 1
    initial_best_qwk = -float("inf")
    scheduler_state   = None

    if os.path.exists(RESUME_PATH):
        print("\nFound Phase 2 checkpoint — resuming.")
        unfreeze_last_blocks(model, num_blocks=2)

        # Same discriminative param groups as the fresh-start branch below —
        # must match in structure so load_full_checkpoint's optimizer
        # state_dict load succeeds.
        backbone_params = [p for n, p in model.named_parameters()
                            if p.requires_grad and n.startswith("backbone")]
        head_params     = [p for n, p in model.named_parameters()
                            if p.requires_grad and not n.startswith("backbone")]

        optimizer = torch.optim.AdamW([
            {"params": backbone_params, "lr": 5e-6},
            {"params": head_params,     "lr": 2e-5},
        ], weight_decay=0.0005)

        loaded_epoch, best_qwk, sched_state = load_full_checkpoint(
            model, optimizer, RESUME_PATH, device
        )
        start_epoch      = loaded_epoch + 1
        initial_best_qwk = best_qwk
        scheduler_state   = sched_state
    else:
        print("\nNo Phase 2 checkpoint — loading best Phase 1 weights to start fresh.")
        model = load_best_model(model, BEST_MODEL_PATH, device)

        # Evaluate the loaded Phase 1 model BEFORE unfreezing/training so the
        # checkpointer knows the real bar to beat — otherwise initial_best_qwk
        # stays at -inf and Phase 2 could overwrite a good Phase 1 model with
        # a worse one on epoch 1 if early fine-tuning is unstable.
        print("Evaluating loaded Phase 1 model to set the Phase 2 baseline...")
        _, p1_metrics = validate(model, val_loader, loss_fn, device, print_report=False)
        initial_best_qwk = p1_metrics["qwk"]
        print(f"  Phase 1 baseline QWK (must beat this to save): {initial_best_qwk:.4f}")

        unfreeze_last_blocks(model, num_blocks=1)

        # Discriminative learning rates, dropped a bit further from last
        # attempt (1e-5/5e-5 -> 5e-6/2e-5), AND unfreezing only the last 2
        # backbone blocks instead of 4 — that's the bigger change: fewer
        # trainable backbone params means less capacity to overfit the
        # 2929-image training set, which is what the train/val loss gap
        # from the last run actually pointed to.
        backbone_params = [p for n, p in model.named_parameters()
                            if p.requires_grad and n.startswith("backbone")]
        head_params     = [p for n, p in model.named_parameters()
                            if p.requires_grad and not n.startswith("backbone")]

        optimizer = torch.optim.AdamW([
            {"params": backbone_params, "lr": 5e-6},
            {"params": head_params,     "lr": 2e-5},
        ], weight_decay=0.0005)

    model_summary(model)

    if start_epoch > EPOCHS:
        print(f"Phase 2 already completed. Skipping to final evaluation.")
    else:
        train(
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
        )

    model = load_best_model(model, BEST_MODEL_PATH, device)
    final_evaluation(model, val_loader, loss_fn, device)


if __name__ == "__main__":
    main()

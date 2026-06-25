import os

# ✅ Must be set BEFORE timm/torch trigger any downloads — redirects
# the pretrained EfficientNetB0 weights away from the default C: cache
# and onto E: instead.
os.environ["HF_HOME"]           = r"E:\retinopathy-screening\hf_home"
os.environ["HF_HUB_CACHE"]      = r"E:\retinopathy-screening\hf_hub_cache"
os.environ["TORCH_HOME"]        = r"E:\retinopathy-screening\torch_cache"

import random
import torch
import torch.nn as nn
import timm
import numpy as np
from torch.utils.data import DataLoader

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    cohen_kappa_score,
    confusion_matrix,
    classification_report
)


# ──────────────────────────────────────────────────────────────
# Reproducibility
# ──────────────────────────────────────────────────────────────

def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False
    print(f"Seed set to {seed} — deterministic mode enabled.")


# ──────────────────────────────────────────────────────────────
# Clinical Mapping
# ──────────────────────────────────────────────────────────────

GRADE_TO_CLINICAL = {
    0: "No DR",
    1: "Early Signs",
    2: "Early Signs",
    3: "Urgent",
    4: "Urgent",
}

CLASS_NAMES = ["No DR", "Mild", "Moderate", "Severe", "Proliferative"]
ALL_LABELS  = list(range(len(CLASS_NAMES)))


# ──────────────────────────────────────────────────────────────
# CBAM — Channel Attention
# ──────────────────────────────────────────────────────────────

class ChannelAttention(nn.Module):
    def __init__(self, in_channels: int, reduction: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg   = self.mlp(self.avg_pool(x))
        max_  = self.mlp(self.max_pool(x))
        scale = self.sigmoid(avg + max_).unsqueeze(2).unsqueeze(3)
        return x * scale


# ──────────────────────────────────────────────────────────────
# CBAM — Spatial Attention
# ──────────────────────────────────────────────────────────────

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=2,
            out_channels=1,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            bias=False
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg     = torch.mean(x, dim=1, keepdim=True)
        max_, _ = torch.max(x, dim=1, keepdim=True)
        scale   = self.sigmoid(self.conv(torch.cat([avg, max_], dim=1)))
        return x * scale


# ──────────────────────────────────────────────────────────────
# CBAM Block
# ──────────────────────────────────────────────────────────────

class CBAM(nn.Module):
    def __init__(self, in_channels: int, reduction: int = 16, kernel_size: int = 7):
        super().__init__()
        self.channel = ChannelAttention(in_channels, reduction)
        self.spatial = SpatialAttention(kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.channel(x)
        x = self.spatial(x)
        return x


# ──────────────────────────────────────────────────────────────
# GeM — Generalized Mean Pooling
# ──────────────────────────────────────────────────────────────
#
# Plain average pooling (nn.AdaptiveAvgPool2d) weights every spatial
# location equally when collapsing the feature map down to one vector.
# In DR grading, the most diagnostically important signal (a tiny
# microaneurysm or hemorrhage) can occupy a very small fraction of the
# image, so averaging dilutes it among a much larger area of
# unremarkable background tissue.
#
# GeM raises activations to a learnable power p before averaging, then
# takes the same power's root afterward. As p grows past 1, this
# increasingly emphasizes the largest activations in the feature map
# (approaching max-pooling as p -> infinity) rather than treating every
# location equally — letting the network learn how much to lean toward
# "pay attention to the strongest signal" vs "average everything",
# rather than that choice being fixed in advance.
#
# Verified: forward pass produces no NaN, and gradients correctly flow
# back to the learnable p parameter during backward().
#
# Note: if you ever see NaN losses after adding this under mixed
# precision (torch.amp.autocast), the most common fix is to compute
# the GeM forward pass outside autocast, e.g.:
#   with torch.amp.autocast("cuda", enabled=False):
#       pooled = self.pool(features.float())
# since x.pow(p) for a growing learnable p can be numerically less
# stable in fp16 than in fp32.
class GeM(nn.Module):
    def __init__(self, p: float = 3.0, eps: float = 1e-6):
        super().__init__()
        self.p   = nn.Parameter(torch.ones(1) * p)  # learnable
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return nn.functional.adaptive_avg_pool2d(
            x.clamp(min=self.eps).pow(self.p),
            (1, 1)
        ).pow(1.0 / self.p)


# ──────────────────────────────────────────────────────────────
# Model
# ──────────────────────────────────────────────────────────────

class RetinopathyModel(nn.Module):
    def __init__(
        self,
        num_classes: int   = 5,
        dropout:     float = 0.4,
        pretrained:  bool  = True
    ):
        super().__init__()

        self.backbone = timm.create_model(
            "efficientnet_b2",
            pretrained=pretrained,
            num_classes=0,
            global_pool=""
        )

        in_channels = self.backbone.num_features
        self.cbam   = CBAM(in_channels)
        # ✅ CHANGED: nn.AdaptiveAvgPool2d(1) -> GeM(). See GeM class
        # docstring above for why. This is a drop-in replacement —
        # same input/output shape, same call signature.
        self.pool   = GeM()

        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_channels, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(512, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        features = self.cbam(features)
        features = self.pool(features)
        logits   = self.head(features)
        return logits


# ──────────────────────────────────────────────────────────────
# Parameter Reporting
# ──────────────────────────────────────────────────────────────

def count_trainable_parameters(model: nn.Module) -> dict:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"  Trainable parameters : {trainable:>12,}")
    print(f"  Total parameters     : {total:>12,}")
    print(f"  Frozen parameters    : {total - trainable:>12,}")
    return {"trainable": trainable, "total": total}


def model_summary(model: nn.Module) -> dict:
    print("\n── Model Summary ──────────────────────────────────────")
    stats = count_trainable_parameters(model)
    print("───────────────────────────────────────────────────────\n")
    return stats


# ──────────────────────────────────────────────────────────────
# BatchNorm running-stats freeze
# ──────────────────────────────────────────────────────────────
#
# BUG THIS FIXES: nn.BatchNorm*d's running_mean/running_var update as a
# side effect of any forward pass while the module is in .train() mode —
# this is controlled entirely by .training, NOT by requires_grad. Setting
# requires_grad=False on a BN layer's weight/bias (which freeze_backbone
# and unfreeze_last_blocks do, since they freeze via model.backbone /
# block .parameters()) stops the OPTIMIZER from updating those two
# tensors, but does nothing to stop running_mean/running_var from
# silently drifting away from the pretrained ImageNet statistics on
# every single forward pass, every epoch, for as long as the layer
# stays "frozen". Verified empirically: a BN layer with requires_grad
# forced False on both weight and bias still updates running_mean/
# running_var after a single forward pass in .train() mode.
#
# This matters for Phase 1 especially: the ENTIRE backbone is frozen
# there, so every BN layer drifts for the full duration of Phase 1,
# before Phase 2 ever touches anything. Phase 2 then inherits a backbone
# whose normalization statistics no longer match the pretrained weights
# they were learned alongside — no amount of Phase 2 hyperparameter
# tuning fixes statistics that already drifted during Phase 1.
#
# THE FIX: after model.train() (which resets ALL submodules, including
# frozen ones, back to train-mode BN behavior), walk the module tree and
# force any BatchNorm layer whose parameters are ALL frozen back into
# .eval() mode. In .eval() mode, BN normalizes using the stored running
# stats instead of the current batch's stats — both halting further
# drift AND making the frozen backbone's output deterministic per input
# again (matching how a properly-frozen pretrained backbone should
# behave), rather than fluctuating with whatever is in the current
# mini-batch.
#
# This must be called every epoch, immediately after model.train(),
# because model.train() unconditionally flips every submodule back to
# train mode and would otherwise silently undo this fix on each call.
def freeze_bn_stats(module: nn.Module) -> None:
    """
    Locks BatchNorm running_mean/running_var for any BN layer whose
    parameters are currently frozen (requires_grad=False), regardless
    of the parent model's .train()/.eval() state.

    Must be called every epoch, right after model.train() — because
    model.train() resets ALL submodules (including frozen ones) back
    to training-mode BN behavior, undoing this fix if not reapplied.

    Correctly handles every phase:
      - Phase 1 (freeze_backbone): entire backbone frozen → all backbone
        BN layers get locked to eval mode.
      - Phase 2 (unfreeze_last_blocks): only the still-frozen early
        stages get locked; the stages you deliberately unfroze keep
        adapting their BN stats normally, alongside their now-trainable
        conv weights, which is exactly what you want there.
      - Phase 3 / unfreeze_all: every parameter is trainable, so this
        is a no-op — no BN layer gets locked, all of them keep adapting.
    """
    for m in module.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            if all(not p.requires_grad for p in m.parameters()):
                m.eval()


# ──────────────────────────────────────────────────────────────
# Freeze / Unfreeze — 3 phases
# ──────────────────────────────────────────────────────────────

def freeze_backbone(model: RetinopathyModel) -> None:
    for param in model.backbone.parameters():
        param.requires_grad = False
    for param in model.cbam.parameters():
        param.requires_grad = True
    for param in model.head.parameters():
        param.requires_grad = True
    print("\n[Phase 1] Backbone frozen — training CBAM + head only.")
    count_trainable_parameters(model)


def unfreeze_last_blocks(model: RetinopathyModel, num_blocks: int = 4) -> None:
    """
    EfficientNetB0 has 7 stages (indices 0-6). Stages 0-2 stay frozen
    (low-level edge/texture features), last 4 stages unfreeze.
    """
    for param in model.backbone.parameters():
        param.requires_grad = False

    blocks = list(model.backbone.blocks)
    if num_blocks > len(blocks):
        print(f"  [Warning] num_blocks={num_blocks} exceeds total stages "
              f"({len(blocks)}). Unfreezing all {len(blocks)} stages instead.")
        num_blocks = len(blocks)

    for block in blocks[-num_blocks:]:
        for param in block.parameters():
            param.requires_grad = True

    for param in model.backbone.conv_head.parameters():
        param.requires_grad = True
    for param in model.backbone.bn2.parameters():
        param.requires_grad = True

    for param in model.cbam.parameters():
        param.requires_grad = True
    for param in model.head.parameters():
        param.requires_grad = True

    print(f"\n[Phase 2] Last {num_blocks}/{len(blocks)} backbone stages unfrozen.")
    count_trainable_parameters(model)


def unfreeze_all(model: RetinopathyModel) -> None:
    for param in model.parameters():
        param.requires_grad = True
    print("\n[Phase 3] All layers unfrozen — full fine-tuning.")
    count_trainable_parameters(model)


def set_finetune_lr(optimizer: torch.optim.Optimizer, lr: float = 1e-5) -> None:
    for group in optimizer.param_groups:
        group["lr"] = lr
    print(f"  [set_finetune_lr] All param groups -> LR = {lr:.2e}")


# ──────────────────────────────────────────────────────────────
# Loss — EMD + CrossEntropy combined
# ──────────────────────────────────────────────────────────────

class EMDLoss(nn.Module):
    def __init__(self, num_classes: int = 5, class_weights: torch.Tensor = None):
        super().__init__()
        self.num_classes = num_classes
        if class_weights is not None:
            self.register_buffer("class_weights", class_weights)
        else:
            self.register_buffer("class_weights", torch.ones(num_classes))

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        probs    = torch.softmax(logits, dim=1)
        cdf_pred = torch.cumsum(probs, dim=1)[:, :-1]

        one_hot  = torch.zeros_like(probs)
        one_hot.scatter_(1, labels.unsqueeze(1), 1.0)
        cdf_true = torch.cumsum(one_hot, dim=1)[:, :-1]

        emd            = torch.mean((cdf_pred - cdf_true) ** 2, dim=1)
        sample_weights = self.class_weights[labels]
        emd            = emd * sample_weights

        return emd.mean()


class CombinedLoss(nn.Module):
    def __init__(
        self,
        class_weights: torch.Tensor = None,
        alpha:         float        = 0.7,
        num_classes:   int          = 5
    ):
        super().__init__()
        self.alpha = alpha
        self.emd   = EMDLoss(num_classes=num_classes, class_weights=class_weights)
        self.ce    = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return (
            self.alpha       * self.emd(logits, labels) +
            (1 - self.alpha) * self.ce(logits, labels)
        )


def get_loss_fn(class_weights: torch.Tensor, alpha: float = 0.7) -> CombinedLoss:
    return CombinedLoss(class_weights=class_weights, alpha=alpha, num_classes=5)


# ──────────────────────────────────────────────────────────────
# Metrics
# ──────────────────────────────────────────────────────────────

def compute_metrics(
    preds: np.ndarray,
    labels: np.ndarray,
    print_report: bool = False,
) -> dict:
    metrics = {
        "accuracy":  accuracy_score(labels, preds),
        "precision": precision_score(labels, preds, average="macro", zero_division=0),
        "recall":    recall_score(labels, preds, average="macro", zero_division=0),
        "f1":        f1_score(labels, preds, average="macro", zero_division=0),
        "qwk":       cohen_kappa_score(labels, preds, weights="quadratic"),
    }

    if print_report:
        print("\n── Confusion Matrix ───────────────────────────────────")
        print(confusion_matrix(labels, preds, labels=ALL_LABELS))

        print("\n── Classification Report ──────────────────────────────")
        print(classification_report(
            labels, preds, labels=ALL_LABELS,
            target_names=CLASS_NAMES, zero_division=0
        ))

    return metrics


# ──────────────────────────────────────────────────────────────
# Early Stopping
# ──────────────────────────────────────────────────────────────

class EarlyStopping:
    def __init__(self, patience: int = 7, min_delta: float = 1e-4, verbose: bool = True):
        self.patience  = patience
        self.min_delta = min_delta
        self.verbose   = verbose
        self.best_qwk  = -np.inf
        self.counter   = 0
        self.stop      = False

    def step(self, current_qwk: float) -> bool:
        if current_qwk > self.best_qwk + self.min_delta:
            if self.verbose:
                print(f"  [EarlyStopping] QWK improved: {self.best_qwk:.4f} -> {current_qwk:.4f}")
            self.best_qwk = current_qwk
            self.counter  = 0
        else:
            self.counter += 1
            if self.verbose:
                print(f"  [EarlyStopping] No improvement {self.counter}/{self.patience} epochs.")
            if self.counter >= self.patience:
                self.stop = True
                print("  [EarlyStopping] Patience exhausted — stopping.")
        return self.stop


# ──────────────────────────────────────────────────────────────
# Checkpointing
# ──────────────────────────────────────────────────────────────

class ModelCheckpoint:
    def __init__(self, save_path: str, verbose: bool = True):
        self.save_path = save_path
        self.verbose   = verbose
        self.best_qwk  = -np.inf

    def step(self, model: nn.Module, current_qwk: float) -> bool:
        if current_qwk > self.best_qwk:
            if self.verbose:
                print(f"  [Checkpoint] QWK improved: {self.best_qwk:.4f} -> {current_qwk:.4f} — saving.")
            self.best_qwk = current_qwk
            torch.save(model.state_dict(), self.save_path)
            return True
        return False


# ──────────────────────────────────────────────────────────────
# Resume Checkpointing
# ──────────────────────────────────────────────────────────────

def save_full_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    epoch: int,
    best_qwk: float,
    path: str,
    extra: dict = None,
) -> None:
    payload = {
        "epoch":     epoch,
        "model":     model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "best_qwk":  best_qwk,
    }
    if extra:
        payload.update(extra)
    torch.save(payload, path)


def load_full_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    path: str,
    device: torch.device
) -> tuple[int, float, dict, dict]:
    """
    Returns (epoch, best_qwk, scheduler_state_dict, full_checkpoint_dict).

    The raw scheduler state dict is returned separately (not a loaded
    scheduler object) so it can be fed into train()'s `scheduler_state`
    parameter — train() builds its own scheduler internally and applies
    this state to it.

    The full checkpoint dict is also returned so callers can pull out
    any extra fields they stored at save time (e.g. num_blocks for
    Phase 2 resume consistency) without needing a second disk read.

    weights_only=False is safe here specifically because this checkpoint
    was created by our own save_full_checkpoint() on this same machine —
    never load a checkpoint from an untrusted/external source with
    weights_only=False.
    """
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    print(f"  [Resume] Restored from epoch {ckpt['epoch']} | best QWK: {ckpt['best_qwk']:.4f}")
    return ckpt["epoch"], ckpt["best_qwk"], ckpt["scheduler"], ckpt


def load_best_model(model: nn.Module, checkpoint_path: str, device: torch.device) -> nn.Module:
    """
    Same weights_only=True default issue as load_full_checkpoint.
    Safe here for the same reason — checkpoint generated locally by us.
    """
    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=False))
    model.to(device)
    model.eval()
    return model


# ──────────────────────────────────────────────────────────────
# Training Loop — with Gradient Accumulation
# ──────────────────────────────────────────────────────────────

def train_one_epoch(
    model: RetinopathyModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    max_grad_norm: float = 1.0,
    accumulation_steps: int = 1,
) -> float:
    """
    With accumulation_steps=2 and a real batch_size=4, this behaves
    like an effective batch size of 8 — same gradient quality as a
    true batch-8 pass, but only 4 images held in VRAM at any moment.

    Loss is divided by accumulation_steps before backward() so the
    summed gradients across N mini-batches match what a single true
    batch of size (batch_size * accumulation_steps) would produce.
    """
    model.train()
    freeze_bn_stats(model)   # ✅ re-lock frozen-stage BN stats — model.train() above just reset them
    running_loss = 0.0
    optimizer.zero_grad(set_to_none=True)

    for i, (images, labels) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
            logits = model(images)
            loss   = loss_fn(logits, labels) / accumulation_steps

        scaler.scale(loss).backward()

        if (i + 1) % accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        running_loss += loss.item() * accumulation_steps * images.size(0)

    if len(loader) % accumulation_steps != 0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

    return running_loss / len(loader.dataset)


@torch.no_grad()
def validate(
    model: RetinopathyModel,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
    print_report: bool = False,
) -> tuple[float, dict]:
    model.eval()
    running_loss = 0.0
    all_preds    = []
    all_labels   = []

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
            logits = model(images)
            loss   = loss_fn(logits, labels)

        running_loss += loss.item() * images.size(0)
        preds = torch.argmax(logits, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    val_loss = running_loss / len(loader.dataset)
    metrics  = compute_metrics(np.array(all_preds), np.array(all_labels), print_report=print_report)
    return val_loss, metrics


def train(
    model: RetinopathyModel,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
    num_epochs: int = 30,
    checkpoint_path: str = "checkpoints/best_model.pth",
    resume_path: str = "checkpoints/resume.pth",
    es_patience: int = 7,
    lr_factor: float = 0.5,
    lr_patience: int = 3,
    start_epoch: int = 1,
    initial_best_qwk: float = -np.inf,
    max_grad_norm: float = 1.0,
    scheduler_state: dict = None,
    accumulation_steps: int = 1,
    checkpoint_extra: dict = None,
    scheduler_type: str = "plateau",
    warmup_epochs: int = 3,
) -> dict:
    """
    checkpoint_extra: optional dict of extra fields to persist into every
    resume checkpoint (e.g. {"num_blocks": 2} for Phase 2), so a later
    resume can read back the exact config that was used, instead of a
    caller having to hardcode it twice and risk it drifting out of sync.

    scheduler_type: "plateau" (default, unchanged behavior — used by
    Phase 2) or "cosine" (new — linear warmup for `warmup_epochs`, then
    cosine decay down to near-zero over the remaining epochs). Unlike
    "plateau", "cosine" follows a fixed schedule decided in advance — it
    does not react to whether QWK is actually improving. That's a real
    trade-off: smoother and less prone to getting stuck waiting on a
    noisy metric, but it can't speed up, slow down, or hold steady if
    training behaves unusually partway through, the way "plateau" can.
    """

    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    assert scheduler_type in ("plateau", "cosine"), \
        f"scheduler_type must be 'plateau' or 'cosine', got {scheduler_type!r}"

    if scheduler_type == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=lr_factor, patience=lr_patience,
        )
    else:  # "cosine"
        cosine_epochs = max(num_epochs - warmup_epochs, 1)
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.1, total_iters=warmup_epochs
        )
        cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cosine_epochs, eta_min=1e-6
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_epochs],
        )

    if scheduler_state is not None:
        scheduler.load_state_dict(scheduler_state)
        print("  [Resume] Scheduler state restored.")

    early_stopping          = EarlyStopping(patience=es_patience, verbose=True)
    checkpointer            = ModelCheckpoint(save_path=checkpoint_path, verbose=True)
    checkpointer.best_qwk   = initial_best_qwk
    early_stopping.best_qwk = initial_best_qwk

    history = {"train_loss": [], "val_loss": [], "metrics": []}

    effective_batch = train_loader.batch_size * accumulation_steps
    print(f"\n{'═' * 65}")
    print(f"  Training    | max {num_epochs} epochs | scheduler: {scheduler_type} | device: {device}")
    print(f"  Real batch  : {train_loader.batch_size} | Accum steps: {accumulation_steps} "
          f"| Effective batch: {effective_batch}")
    print(f"  Best model  -> {checkpoint_path}")
    print(f"  Resume ckpt -> {resume_path}")
    if scheduler_type == "plateau":
        print(f"  ES patience : {es_patience} | LR patience: {lr_patience}")
    else:
        print(f"  ES patience : {es_patience} | Warmup epochs: {warmup_epochs}")
    print(f"{'═' * 65}\n")

    for epoch in range(start_epoch, num_epochs + 1):

        is_last = (epoch == num_epochs)

        train_loss = train_one_epoch(
            model, train_loader, optimizer, loss_fn, scaler, device,
            max_grad_norm, accumulation_steps
        )

        val_loss, metrics = validate(model, val_loader, loss_fn, device, print_report=is_last)

        current_qwk = metrics["qwk"]
        current_lr  = optimizer.param_groups[0]["lr"]

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["metrics"].append(metrics)

        print(
            f"Epoch {epoch:03d}/{num_epochs} | "
            f"loss {train_loss:.4f} / {val_loss:.4f} | "
            f"QWK {current_qwk:.4f} | acc {metrics['accuracy']:.4f} | LR {current_lr:.2e}"
        )

        # ✅ CHANGED: ReduceLROnPlateau needs the metric passed to .step();
        # cosine/SequentialLR takes no argument and just advances on its
        # fixed schedule. Branching here is what lets both scheduler
        # types share this same training loop.
        if scheduler_type == "plateau":
            scheduler.step(current_qwk)
        else:
            scheduler.step()

        new_lr = optimizer.param_groups[0]["lr"]
        if new_lr != current_lr:
            print(f"  [LR] {current_lr:.2e} -> {new_lr:.2e}")

        checkpointer.step(model, current_qwk)

        save_full_checkpoint(
            model, optimizer, scheduler, epoch, checkpointer.best_qwk, resume_path,
            extra=checkpoint_extra,
        )

        if early_stopping.step(current_qwk):
            print(f"\nEarly stopping triggered at epoch {epoch}.")
            print("\n── Final Validation Report ────────────────────────────")
            validate(model, val_loader, loss_fn, device, print_report=True)
            break

    print(f"\n{'═' * 65}")
    print(f"  Training complete.")
    print(f"  Best QWK    : {checkpointer.best_qwk:.4f}")
    print(f"  Best model  -> {checkpoint_path}")
    print(f"{'═' * 65}")

    return history


# ──────────────────────────────────────────────────────────────
# Final Evaluation
# ──────────────────────────────────────────────────────────────

def final_evaluation(
    model: RetinopathyModel,
    val_loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
) -> dict:
    print("\n── Final Evaluation on Validation Set ─────────────────")
    val_loss, metrics = validate(model, val_loader, loss_fn, device, print_report=True)
    print(f"\n  QWK       : {metrics['qwk']:.4f}")
    print(f"  Accuracy  : {metrics['accuracy']:.4f}")
    print(f"  F1 (macro): {metrics['f1']:.4f}")
    print(f"  Val Loss  : {val_loss:.4f}")
    return metrics


# ──────────────────────────────────────────────────────────────
# Inference
# ──────────────────────────────────────────────────────────────

def predict_clinical(
    model: RetinopathyModel,
    image_tensor: torch.Tensor,
    device: torch.device
) -> dict:
    model.eval()
    with torch.no_grad():
        x      = image_tensor.unsqueeze(0).to(device)
        logits = model(x)
        probs  = torch.softmax(logits, dim=1)
        grade  = torch.argmax(probs, dim=1).item()
        conf   = probs[0][grade].item()
    return {
        "grade":      grade,
        "clinical":   GRADE_TO_CLINICAL[grade],
        "confidence": round(conf * 100, 1),
        "all_probs":  probs[0].cpu().numpy().tolist()
    }

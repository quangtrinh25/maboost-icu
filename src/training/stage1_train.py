"""
src/training/stage1_train.py
=============================
Stage 1: joint mortality + LOS pre-training.

Based on APRICOT-Mamba (Contreras et al. 2024) training protocol:
- Class weights applied to loss for class imbalance (paper §2.4)
- Adam optimiser (paper uses standard Adam/AdamW)
- Pure float32 — mamba_simple.py has no fp32 accumulators
- NaN gradient hooks on Mamba parameters (mandatory for mamba_simple.py)
- EMA checkpointing for better generalisation
- Focal loss (γ=2) improves on paper's weighted CE for 16% positive rate
- Log-cosh LOS loss robust to 30+ day outliers
- Mixup augmentation for synthetic positives

Benchmark note
--------------
GRUDImputer lives inside MambaEncoder by design (Stage 1 wants end-to-end
gradients through imputation).  This means the raw DataLoader batches still
contain NaN.  LSTM / Transformer benchmarks that don't share this encoder
will crash on NaN inputs unless a shared imputation step runs upstream in
the ETL/pipeline before the DataLoader is constructed.
"""
from __future__ import annotations
import copy
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from mamba_ssm import Mamba
from src.models.mamba_encoder import DualHeadMamba, _register_nan_hooks


# ---------------------------------------------------------------------------
# Losses
# ---------------------------------------------------------------------------
AUX_WEIGHT = 0.1
class FocalLoss(nn.Module):
    """FL = −α_t (1−p_t)^γ log(p_t), γ=2 (Lin et al. 2017)"""
    def __init__(self, alpha: torch.Tensor, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor,
                targets: torch.Tensor) -> torch.Tensor:
        ce   = F.cross_entropy(logits, targets, reduction="none")
        pt   = torch.exp(-ce)
        loss = self.alpha[targets] * (1.0 - pt) ** self.gamma * ce
        return loss.mean()


def log_cosh_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Quadratic near 0, MAE for large errors — robust to LOS outliers."""
    d = pred - target
    return torch.mean(d + F.softplus(-2.0 * d) - np.log(2.0))


# ---------------------------------------------------------------------------
# Mixup
# ---------------------------------------------------------------------------

def mixup_batch(x, tau, mask, y_mort, y_los, x_static=None, alpha=0.4):
    """
    Append λ-interpolated (pos, random_neg) pairs. λ~Beta[0.3,0.7].

    x_static is mixed alongside x so the encoder's static branch receives
    a consistently sized batch (shape mismatch causes a forward-pass crash
    if x_static is omitted from the mix).
    """
    pos = (y_mort == 1).nonzero(as_tuple=False).squeeze(-1)
    neg = (y_mort == 0).nonzero(as_tuple=False).squeeze(-1)
    if pos.numel() == 0 or neg.numel() == 0:
        return x, tau, mask, y_mort, y_los, x_static

    n   = pos.numel()
    lam = torch.distributions.Beta(alpha, alpha).sample((n,)).to(x.device).clamp(0.3, 0.7)
    rn  = neg[torch.randint(neg.numel(), (n,))]
    l3  = lam.view(-1, 1, 1)

    xs_out = None
    if x_static is not None:
        l1     = lam.view(-1, 1)
        xs_out = torch.cat([x_static,
                            l1 * x_static[pos] + (1 - l1) * x_static[rn]])

    return (
        torch.cat([x,      l3             * x[pos]    + (1 - l3)             * x[rn]]),
        torch.cat([tau,    lam.view(-1,1) * tau[pos]  + (1 - lam.view(-1,1)) * tau[rn]]),
        torch.cat([mask,   l3             * mask[pos] + (1 - l3)             * mask[rn]]),
        torch.cat([y_mort, torch.ones(n, dtype=torch.long, device=x.device)]),
        torch.cat([y_los,  lam * y_los[pos] + (1 - lam) * y_los[rn]]),
        xs_out,
    )


# ---------------------------------------------------------------------------
# EMA
# ---------------------------------------------------------------------------

class ModelEMA:
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay  = decay
        self.shadow = copy.deepcopy(model.state_dict())

    @torch.no_grad()
    def update(self, model: nn.Module):
        for k, v in model.state_dict().items():
            if v.dtype.is_floating_point:
                self.shadow[k] = self.shadow[k] * self.decay + v.float() * (1 - self.decay)

    def apply(self, model: nn.Module):
        device = next(model.parameters()).device
        model.load_state_dict(
            {k: v.to(device) for k, v in self.shadow.items()}
        )


# ---------------------------------------------------------------------------
# LR scheduler
# ---------------------------------------------------------------------------

def _warmup_cosine(step: int, total: int, warmup: int,
                   min_r: float = 0.01) -> float:
    if step < warmup:
        return float(step) / max(warmup, 1)
    p = (step - warmup) / max(total - warmup, 1)
    return min_r + (1.0 - min_r) * 0.5 * (1.0 + np.cos(np.pi * p))


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def _eval(model: DualHeadMamba, loader: DataLoader, device: str) -> float:
    model.eval()
    probs, labels = [], []
    for batch in loader:
        x, tau, mask = (batch[i].to(device) for i in range(3))
        x_static     = batch[3].to(device) if len(batch) == 6 else None
        logits, _    = model(x, tau, mask, x_static=x_static)
        probs.append(torch.softmax(logits.float(), -1)[:, 1].cpu())
        labels.append(batch[-2].cpu())

    p = np.nan_to_num(torch.cat(probs).numpy(),  nan=0.5, posinf=1.0, neginf=0.0)
    l = np.nan_to_num(torch.cat(labels).numpy(), nan=0.0).astype(int)
    p = np.clip(p, 1e-7, 1.0 - 1e-7)
    if l.sum() == 0 or (1 - l).sum() == 0:
        return 0.5
    return float(roc_auc_score(l, p))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def train_stage1(
    model:        DualHeadMamba,
    train_loader: DataLoader,
    val_loader:   DataLoader,
    epochs:       int   = 100,
    lr:           float = 1e-3,
    patience:     int   = 15,
    clip_norm:    float = 1.0,
    ckpt_dir:     str   = "checkpoints",
    device:       str   = "cuda",
    warmup_frac:  float = 0.10,
    ema_decay:    float = 0.999,
    focal_gamma:  float = 2.0,
    loss_weight:  float = 0.5,
    use_mixup:    bool  = True,
    mixup_alpha:  float = 0.4,
) -> DualHeadMamba:
    Path(ckpt_dir).mkdir(parents=True, exist_ok=True)
    ckpt  = Path(ckpt_dir) / "encoder_best.pth"
    model = model.to(device)

    # ── NaN gradient hooks — mandatory for mamba_simple.py ────────────────
    _register_nan_hooks(model)

    # ── Weight init — skip Mamba internals ────────────────────────────────
    # Mamba's dt_proj.bias is carefully set to keep exp(Δ·A) stable.
    # Overwriting with xavier/zeros destroys this → NaN from first forward.
    mamba_ids = {id(p) for m in model.modules()
                 if isinstance(m, Mamba) for p in m.parameters()}
    for m in model.modules():
        if isinstance(m, Mamba):
            continue
        if isinstance(m, (nn.Linear, nn.Conv1d)):
            if any(id(p) in mamba_ids for p in m.parameters()):
                continue
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    # ── Class weights (paper §2.4) ─────────────────────────────────────
    ds    = train_loader.dataset
    n_tot = len(ds)
    if hasattr(ds, "mort") and hasattr(ds, "ids"):
        n_pos = sum(1 for s in ds.ids if ds.mort.get(s, 0) == 1)
    else:
        # Sample up to 5000 examples to estimate positive rate
        n_sample = min(n_tot, 5000)
        n_pos    = sum(int(ds[i][-2].item()) for i in range(n_sample))
        if n_tot > n_sample:
            n_pos = int(n_pos * n_tot / n_sample)
    n_neg = n_tot - n_pos
    print(f"[Stage 1] pos={n_pos:,} neg={n_neg:,} ({100*n_pos/max(n_tot,1):.1f}%)")

    # alpha[0] = weight for negative class, alpha[1] = weight for positive
    w_neg = 1.0 / max(n_neg, 1)
    w_pos = 1.0 / max(n_pos, 1)
    total = w_neg + w_pos
    alpha = torch.tensor([w_neg / total,
                        w_pos / total], device=device)
    focal = FocalLoss(alpha=alpha, gamma=focal_gamma)

    # ── Optimiser + LR ────────────────────────────────────────────────────
    opt = torch.optim.AdamW(model.parameters(), lr=lr,
                            weight_decay=1e-4, betas=(0.9, 0.98))
    total_steps  = epochs * len(train_loader)
    warmup_steps = int(total_steps * warmup_frac)
    scheduler    = torch.optim.lr_scheduler.LambdaLR(
        opt, lr_lambda=lambda s: _warmup_cosine(s, total_steps, warmup_steps)
    )

    print(f"[Stage 1] float32 | mixup={use_mixup} | γ={focal_gamma} | "
          f"warmup={warmup_frac:.0%} | nan_hooks=ON")

    ema      = ModelEMA(model, decay=ema_decay)
    best_auc = 0.0
    no_imp   = 0
    print(f"[Stage 1] Training — up to {epochs} epochs  patience={patience}")

    for ep in range(1, epochs + 1):
        model.train()
        ep_loss = ep_lm = ep_ll = ep_grad = 0.0
        n_ok = n_skip = 0

        for batch in train_loader:
            x, tau, mask = (batch[i].to(device) for i in range(3))
            has_static   = len(batch) == 6
            x_static     = batch[3].to(device) if has_static else None
            y_mort       = batch[-2].to(device)
            y_los        = batch[-1].to(device)

            if use_mixup:
                x, tau, mask, y_mort, y_los, x_static = mixup_batch(
                    x, tau, mask, y_mort, y_los,
                    x_static=x_static, alpha=mixup_alpha,
                )

            opt.zero_grad(set_to_none=True)

            # Pure float32 — no autocast
            logits, pred_los = model(x, tau, mask, x_static=x_static)
            logits   = logits.clamp(-20.0, 20.0)
            pred_los = pred_los.clamp(-8.0,  8.0)
            loss_m   = focal(logits, y_mort)
            loss_l   = log_cosh_loss(pred_los, torch.log1p(y_los))
            loss     = loss_m + loss_weight * loss_l

            # Guard: skip batch if loss is non-finite (NaN grad corrupts Adam m2)
            if not torch.isfinite(loss):
                opt.zero_grad(set_to_none=True)
                n_skip += 1
                continue

            loss.backward()

            # Double guard: zero any NaN gradients that slipped past hooks
            for p in model.parameters():
                if p.grad is not None:
                    torch.nan_to_num_(p.grad, nan=0.0, posinf=0.0, neginf=0.0)

            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
            if not torch.isfinite(grad_norm):
                opt.zero_grad(set_to_none=True)
                n_skip += 1
                continue

            opt.step()
            scheduler.step()
            ema.update(model)

            ep_loss += loss.item()
            ep_lm   += loss_m.item()
            ep_ll   += loss_l.item()
            ep_grad += float(grad_norm)
            n_ok    += 1

        # Validate with EMA weights (swap in, eval, swap back)
        saved_state = copy.deepcopy(model.state_dict())
        ema.apply(model)
        auc = _eval(model, val_loader, device)
        model.load_state_dict(saved_state)

        nb = max(n_ok, 1)
        print(
            f"  ep {ep:3d}/{epochs} | "
            f"loss={ep_loss/nb:.4f} "
            f"(m={ep_lm/nb:.4f} l={ep_ll/nb:.4f}) | "
            f"grad={ep_grad/nb:.3f} | AUC={auc:.4f}"
            + (f" [{n_skip} skip]" if n_skip else "")
        )

        if auc > best_auc + 1e-4:
            best_auc, no_imp = auc, 0
            # Save EMA encoder weights
            ema.apply(model)
            torch.save(model.encoder.state_dict(), ckpt)
            model.load_state_dict(saved_state)
            print(f"    ✓ AUC {best_auc:.4f} — saved")
        else:
            no_imp += 1
            if no_imp >= patience:
                print(f"  Early stop at epoch {ep}")
                break

    # Restore best encoder weights into model before returning
    model.encoder.load_state_dict(
        torch.load(ckpt, map_location=device, weights_only=True)
    )
    print(f"[Stage 1] Done. Best val AUC: {best_auc:.4f}")
    return model
"""
src/training/stage1_train.py
=============================
Stage 1: joint mortality + LOS pre-training.

v7 change: added optional auxiliary reconstruction loss from SCI.
  - Helps SCI kernel κ learn meaningful interpolation
  - aux_weight=0.1 (default) — tune down to 0.05 if unstable
  - Disable with aux_weight=0.0

Fix: _aux_reconstruction_loss: B,T,F → B,T,n_feat (avoids F shadow).
All function names and signatures unchanged.
"""
from __future__ import annotations
import copy
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader

from mamba_ssm import Mamba
from src.models.mamba_encoder import DualHeadMamba, _register_nan_hooks


# ---------------------------------------------------------------------------
# Losses
# ---------------------------------------------------------------------------
class FocalLoss(nn.Module):
    def __init__(self, alpha: torch.Tensor, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        ce   = F.cross_entropy(logits, targets, reduction="none")
        pt   = torch.exp(-ce)
        loss = self.alpha[targets] * (1.0 - pt) ** self.gamma * ce
        return loss.mean()


def log_cosh_loss(pred, target):
    d = pred - target
    return torch.mean(d + F.softplus(-2.0 * d) - np.log(2.0))


# ---------------------------------------------------------------------------
# Mixup
# ---------------------------------------------------------------------------

def mixup_batch(x, tau, mask, y_mort, y_los, x_static=None, alpha=0.4):
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
        model.load_state_dict({k: v.to(device) for k, v in self.shadow.items()})


# ---------------------------------------------------------------------------
# LR scheduler
# ---------------------------------------------------------------------------

def _warmup_cosine(step, total, warmup, min_r=0.01):
    if step < warmup:
        return float(step) / max(warmup, 1)
    p = (step - warmup) / max(total - warmup, 1)
    return min_r + (1.0 - min_r) * 0.5 * (1.0 + np.cos(np.pi * p))


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def _eval_metrics(model: DualHeadMamba, loader: DataLoader, device: str) -> dict:
    model.eval()
    probs, labels = [], []
    for batch in loader:
        x, tau, mask = (batch[i].to(device) for i in range(3))
        # IMPORTANT: Stage 1 saves only model.encoder.state_dict(), not the
        # temporary static branch or prediction heads. Checkpoint selection must
        # therefore score encoder quality directly, without x_static.
        logits, _    = model(x, tau, mask, x_static=None)
        probs.append(torch.softmax(logits.float(), -1)[:, 1].cpu())
        labels.append(batch[-2].cpu())

    p = np.nan_to_num(torch.cat(probs).numpy(),  nan=0.5, posinf=1.0, neginf=0.0)
    l = np.nan_to_num(torch.cat(labels).numpy(), nan=0.0).astype(int)
    p = np.clip(p, 1e-7, 1.0 - 1e-7)
    if l.sum() == 0 or (1 - l).sum() == 0:
        base = float(l.mean()) if len(l) > 0 else 0.0
        return {"auroc": 0.5, "auprc": base}
    return {
        "auroc": float(roc_auc_score(l, p)),
        "auprc": float(average_precision_score(l, p)),
    }


# ---------------------------------------------------------------------------
# Auxiliary reconstruction loss helper
# ---------------------------------------------------------------------------

def _aux_reconstruction_loss(
    model:      DualHeadMamba,
    x:          torch.Tensor,
    tau:        torch.Tensor,
    mask:       torch.Tensor,
    held_rate:  float = 0.2,
) -> torch.Tensor:
    """
    Auxiliary SCI reconstruction loss.
    Hold out 20% of observed positions, ask SCI+CCI to reconstruct them.
    Fix: use n_feat instead of F to avoid shadowing torch.nn.functional.
    """
    held_mask = (mask * (torch.rand_like(mask) < held_rate)).float()
    mask_reduced = mask * (1.0 - held_mask)

    enc = model.encoder

    if mask_reduced.dim() == 2:
        m3 = mask_reduced.unsqueeze(-1).expand_as(x)
    else:
        m3 = mask_reduced
    x_hat = enc.imputer(x, tau, m3)

    sigma, lam, gamma = enc.sci(x_hat, m3, tau)
    chi, _ = enc.cci(sigma, lam, gamma)

    # FIX: n_feat instead of F to avoid shadowing F = nn.functional
    B, T, n_feat = x.shape
    device  = x.device
    kappa   = torch.nn.functional.softplus(enc.sci.log_kernel)  # (n_feat,)

    t_hours = tau.cumsum(dim=1) / 3600.0
    ref_t   = torch.linspace(0, enc.sci.window_hours, enc.sci.ref_points, device=device)

    dist2 = (t_hours.unsqueeze(2) - ref_t.unsqueeze(0).unsqueeze(0)) ** 2  # (B,T,R)
    w_back = torch.exp(-kappa.view(1, 1, 1, n_feat) *
                       dist2.unsqueeze(-1).expand(B, T, enc.sci.ref_points, n_feat))
    w_sum  = w_back.sum(dim=2).clamp(min=1)
    x_recon = (w_back * chi.unsqueeze(1).expand(B, T, enc.sci.ref_points, n_feat)
               ).sum(dim=2) / w_sum

    if held_mask.dim() == 2:
        hm3 = held_mask.unsqueeze(-1).expand_as(x)
    else:
        hm3 = held_mask

    x_safe = torch.nan_to_num(x, nan=0.0)
    loss   = ((x_safe - x_recon) ** 2) * hm3
    n      = hm3.sum().clamp(min=1)
    return loss.sum() / n


# ---------------------------------------------------------------------------
# Main — train_stage1
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
    oversample_pos: bool = False,
    aux_weight:   float = 0.1,
    aux_held_rate: float = 0.2,
    remaining_los_weight: float = 0.3,
    trend_stop_window: int = 8,
    trend_stop_delta: float = 0.01,
    selection_metric: str = "auroc",
) -> DualHeadMamba:
    Path(ckpt_dir).mkdir(parents=True, exist_ok=True)
    ckpt  = Path(ckpt_dir) / "encoder_best.pth"
    model = model.to(device)

    _register_nan_hooks(model)

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

    ds    = train_loader.dataset
    n_tot = len(ds)
    if hasattr(ds, "mort") and hasattr(ds, "ids"):
        n_pos = sum(1 for s in ds.ids if ds.mort.get(s, 0) == 1)
    else:
        n_sample = min(n_tot, 5000)
        n_pos    = sum(int(ds[i][-2].item()) for i in range(n_sample))
        if n_tot > n_sample:
            n_pos = int(n_pos * n_tot / n_sample)
    n_neg = n_tot - n_pos
    print(f"[Stage 1] pos={n_pos:,} neg={n_neg:,} ({100*n_pos/max(n_tot,1):.1f}%)")

    if oversample_pos:
        alpha = torch.tensor([0.5, 0.5], device=device)
    else:
        w_neg = 1.0 / max(n_neg, 1)
        w_pos = 1.0 / max(n_pos, 1)
        total = w_neg + w_pos
        alpha = torch.tensor([w_neg / total, w_pos / total], device=device)
    focal = FocalLoss(alpha=alpha, gamma=focal_gamma)

    opt = torch.optim.AdamW(model.parameters(), lr=lr,
                            weight_decay=1e-4, betas=(0.9, 0.98))
    total_steps  = epochs * len(train_loader)
    warmup_steps = int(total_steps * warmup_frac)
    scheduler    = torch.optim.lr_scheduler.LambdaLR(
        opt, lr_lambda=lambda s: _warmup_cosine(s, total_steps, warmup_steps)
    )

    use_event_tokens = bool(getattr(model.encoder, "event_token_mode", False))
    use_aux = aux_weight > 0.0 and not use_event_tokens
    selection_metric = str(selection_metric).lower().strip()
    if selection_metric not in {"auroc", "auprc"}:
        selection_metric = "auroc"

    print(f"[Stage 1] float32 | mixup={use_mixup} | oversample={oversample_pos} | γ={focal_gamma} | "
          f"warmup={warmup_frac:.0%} | nan_hooks=ON | "
          f"aux_loss={'OFF(event_tokens)' if (aux_weight > 0.0 and use_event_tokens) else ('ON w=' + str(aux_weight) if use_aux else 'OFF')} | "
          f"select={selection_metric.upper()}")

    ema      = ModelEMA(model, decay=ema_decay)
    best_score = -np.inf
    no_imp   = 0
    score_hist = []
    print(f"[Stage 1] Training — up to {epochs} epochs  patience={patience}")

    for ep in range(1, epochs + 1):
        model.train()
        ep_loss = ep_lm = ep_ll = ep_aux = ep_grad = 0.0
        ep_lr = 0.0
        n_ok = n_skip = 0

        for batch in train_loader:
            x, tau, mask = (batch[i].to(device) for i in range(3))
            has_static = len(batch) in (6, 7)
            has_rem = len(batch) in (6, 7) and model.enable_remaining_head
            if len(batch) == 7:
                x_static = batch[3].to(device)
                y_los_rem = batch[4].to(device)
                y_mort = batch[5].to(device)
                y_los = batch[6].to(device)
            elif len(batch) == 6 and has_static:
                x_static = batch[3].to(device)
                y_los_rem = None
                y_mort = batch[4].to(device)
                y_los = batch[5].to(device)
            elif len(batch) == 6:
                x_static = None
                y_los_rem = batch[3].to(device)
                y_mort = batch[4].to(device)
                y_los = batch[5].to(device)
            else:
                x_static = None
                y_los_rem = None
                y_mort = batch[-2].to(device)
                y_los = batch[-1].to(device)

            if use_mixup:
                x, tau, mask, y_mort, y_los, x_static = mixup_batch(
                    x, tau, mask, y_mort, y_los,
                    x_static=x_static, alpha=mixup_alpha,
                )
                y_los_rem = None

            opt.zero_grad(set_to_none=True)

            if model.enable_remaining_head:
                logits, pred_los, pred_los_rem = model.forward_research(
                    x, tau, mask, x_static=x_static
                )
            else:
                logits, pred_los = model(x, tau, mask, x_static=x_static)
                pred_los_rem = None
            logits   = logits.clamp(-20.0, 20.0)
            pred_los = pred_los.clamp(-8.0, 8.0)
            loss_m   = focal(logits, y_mort)
            loss_l   = log_cosh_loss(pred_los, torch.log1p(y_los))
            loss     = loss_m + loss_weight * loss_l
            if pred_los_rem is not None and y_los_rem is not None:
                pred_los_rem = pred_los_rem.clamp(-8.0, 8.0)
                loss_lr = log_cosh_loss(pred_los_rem, torch.log1p(y_los_rem))
                loss = loss + remaining_los_weight * loss_lr
                ep_lr += loss_lr.item()

            if use_aux:
                try:
                    mask_3d = mask if mask.dim() == 3 else mask.unsqueeze(-1).expand_as(x)
                    loss_aux = _aux_reconstruction_loss(
                        model, x, tau, mask_3d, held_rate=aux_held_rate
                    )
                    if torch.isfinite(loss_aux):
                        loss = loss + aux_weight * loss_aux
                        ep_aux += loss_aux.item()
                except Exception:
                    pass

            if not torch.isfinite(loss):
                opt.zero_grad(set_to_none=True)
                n_skip += 1
                continue

            loss.backward()

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

        saved_state = copy.deepcopy(model.state_dict())
        ema.apply(model)
        val_metrics = _eval_metrics(model, val_loader, device)
        auroc = float(val_metrics["auroc"])
        auprc = float(val_metrics["auprc"])
        score = auprc if selection_metric == "auprc" else auroc
        score_hist.append(float(score))
        model.load_state_dict(saved_state)

        nb = max(n_ok, 1)
        aux_str = f" aux={ep_aux/nb:.4f}" if use_aux else ""
        lr_str = f" rem={ep_lr/nb:.4f}" if model.enable_remaining_head else ""
        print(
            f"  ep {ep:3d}/{epochs} | "
            f"loss={ep_loss/nb:.4f} "
            f"(m={ep_lm/nb:.4f} l={ep_ll/nb:.4f}{lr_str}{aux_str}) | "
            f"grad={ep_grad/nb:.3f} | AUROC={auroc:.4f} | AUPRC={auprc:.4f}"
            + (f" [{n_skip} skip]" if n_skip else "")
        )

        if score > best_score + 1e-4:
            best_score, no_imp = score, 0
            ema.apply(model)
            torch.save(model.encoder.state_dict(), ckpt)
            model.load_state_dict(saved_state)
            print(f"    ✓ {selection_metric.upper()} {best_score:.4f} — saved")
        else:
            no_imp += 1
            if trend_stop_window > 1 and len(score_hist) >= trend_stop_window:
                recent = score_hist[-trend_stop_window:]
                head = recent[: trend_stop_window // 2]
                tail = recent[trend_stop_window // 2 :]
                if len(head) > 0 and len(tail) > 0:
                    if (float(np.mean(head)) - float(np.mean(tail))) >= trend_stop_delta:
                        print(
                            f"  Trend-stop at epoch {ep}: recent {selection_metric.upper()} drop "
                            f"{float(np.mean(head)) - float(np.mean(tail)):.4f} "
                            f"(window={trend_stop_window})"
                        )
                        break
            if no_imp >= patience:
                print(f"  Early stop at epoch {ep}")
                break

    model.encoder.load_state_dict(
        torch.load(ckpt, map_location=device, weights_only=True)
    )
    print(f"[Stage 1] Done. Best val {selection_metric.upper()}: {best_score:.4f}")
    return model

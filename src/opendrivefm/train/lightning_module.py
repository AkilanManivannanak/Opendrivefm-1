from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Tuple

import torch
import torch.nn.functional as F
import pytorch_lightning as pl

from opendrivefm.models.model import OpenDriveFM


def _dl_kwargs() -> dict[str, Any]:
    return {"num_workers": 0, "pin_memory": False, "persistent_workers": False}


def dice_loss_from_logits(logits: torch.Tensor, target01: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    prob = torch.sigmoid(logits)
    prob = prob.view(prob.size(0), -1)
    tgt = target01.view(target01.size(0), -1)
    inter = (prob * tgt).sum(dim=1)
    denom = prob.sum(dim=1) + tgt.sum(dim=1)
    dice = (2.0 * inter + eps) / (denom + eps)
    return 1.0 - dice.mean()


def ade_fde(pred: torch.Tensor, gt: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    pred, gt: (B,T,2)
    returns:
      ade: (B,)
      fde: (B,)
    """
    d = torch.linalg.norm(pred - gt, dim=-1)  # (B,T)
    return d.mean(dim=1), d[:, -1]


@dataclass
class LossCfg:
    # occupancy
    occ_dice_w: float = 0.7
    pos_weight_cap: float = 15.0

    # traj
    traj_beta: float = 1.0     # SmoothL1 beta
    traj_w: float = 1.0        # weight on traj loss
    resid_l2_w: float = 0.02   # keep residual small (important when CV is already strong)
    time_weight_power: float = 1.0  # >1 weights later steps more


class LitOpenDriveFM(pl.LightningModule):
    """
    Tesla-style: traj = CV_prior(motion, t_rel) + residual(images)

    This makes your model instantly competitive because:
      - Your labels are close to constant-velocity already (CV baseline is strong)
      - The network only learns corrections, not physics
    """

    def __init__(
        self,
        lr: float = 3e-4,
        d: int = 384,
        bev: int = 64,
        horizon: int = 12,
        loss: LossCfg | None = None,
        weight_decay: float = 1e-2,
        grad_clip: float = 1.0,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["loss"])
        self.model = OpenDriveFM(d=d, bev_h=bev, bev_w=bev, horizon=horizon)

        self.lr = lr
        self.weight_decay = weight_decay
        self.grad_clip = grad_clip
        self.loss_cfg = loss or LossCfg()
        self.horizon = int(horizon)

    def forward(self, x: torch.Tensor):
        return self.model(x)  # expected (occ_logits, traj_head)

    def _occ_loss(self, occ_logits: torch.Tensor, occ_tgt: torch.Tensor) -> torch.Tensor:
        # Dynamic pos_weight per-batch to handle class imbalance robustly
        # pos_weight = neg/pos (capped)
        with torch.no_grad():
            pos = occ_tgt.sum()
            total = torch.tensor(float(occ_tgt.numel()), device=occ_tgt.device, dtype=occ_tgt.dtype)
            neg = total - pos
            pw = (neg / (pos + 1e-6)).clamp(min=1.0, max=float(self.loss_cfg.pos_weight_cap))
        bce = F.binary_cross_entropy_with_logits(occ_logits, occ_tgt, pos_weight=pw)
        dsc = dice_loss_from_logits(occ_logits, occ_tgt)
        return bce + self.loss_cfg.occ_dice_w * dsc

    def _make_cv_traj(self, motion: torch.Tensor, t_rel: torch.Tensor) -> torch.Tensor:
        """
        motion: (B,3) = [dt_prev, vx, vy]
        t_rel:  (B,T) seconds
        returns cv_traj: (B,T,2)
        """
        dt_prev = motion[:, 0:1]                 # (B,1)
        vxy = motion[:, 1:3]                     # (B,2)

        # If dt_prev==0, treat velocity as unknown -> force prior to 0
        valid = (dt_prev > 0.0).to(vxy.dtype)    # (B,1)
        vxy = vxy * valid

        cv = t_rel.unsqueeze(-1) * vxy.unsqueeze(1)  # (B,T,2)
        return cv

    def _traj_residual_loss(
        self,
        traj_res: torch.Tensor,   # (B,T,2)
        traj_t: torch.Tensor,     # (B,T,2)
        cv_traj: torch.Tensor,    # (B,T,2)
        t_rel: torch.Tensor,      # (B,T)
    ) -> torch.Tensor:
        # residual target
        target_res = traj_t - cv_traj

        # time weighting: later steps matter more
        # normalize by last time to keep stable
        tnorm = t_rel / (t_rel[:, -1:].clamp(min=1e-6))
        w = (tnorm.clamp(min=0.0) ** float(self.loss_cfg.time_weight_power)).unsqueeze(-1)  # (B,T,1)

        per = F.smooth_l1_loss(traj_res, target_res, beta=self.loss_cfg.traj_beta, reduction="none")  # (B,T,2)
        per = per * w
        loss = per.mean()

        # regularize residual magnitude (prevents overfitting when CV is already excellent)
        l2 = (traj_res ** 2).mean()
        return loss + self.loss_cfg.resid_l2_w * l2

    def _unpack_batch(self, batch):
        """
        Supports:
          (x, occ, traj)                         -> no motion, no t_rel
          (x, occ, traj, motion)                 -> motion only (t_rel assumed 0.5s steps)
          (x, occ, traj, motion, t_rel)          -> motion + t_rel (preferred)
        """
        if not isinstance(batch, (tuple, list)):
            raise TypeError("Batch must be a tuple/list from DataLoader.")

        if len(batch) == 3:
            x, occ_t, traj_t = batch
            B, T = traj_t.shape[0], traj_t.shape[1]
            motion = torch.zeros((B, 3), device=traj_t.device, dtype=traj_t.dtype)
            t_rel = (torch.arange(1, T + 1, device=traj_t.device, dtype=traj_t.dtype)[None, :]).repeat(B, 1) * 0.5
            return x, occ_t, traj_t, motion, t_rel

        if len(batch) == 4:
            x, occ_t, traj_t, motion = batch
            B, T = traj_t.shape[0], traj_t.shape[1]
            t_rel = (torch.arange(1, T + 1, device=traj_t.device, dtype=traj_t.dtype)[None, :]).repeat(B, 1) * 0.5
            return x, occ_t, traj_t, motion, t_rel

        if len(batch) == 5:
            x, occ_t, traj_t, motion, t_rel = batch
            return x, occ_t, traj_t, motion, t_rel

        raise ValueError(f"Unexpected batch length: {len(batch)}")

    def _step(self, batch):
        x, occ_t, traj_t, motion, t_rel = self._unpack_batch(batch)

        occ_logits, traj_head = self(x)

        # shapes
        if occ_t.ndim == 3:
            occ_t = occ_t.unsqueeze(1)
        if occ_logits.ndim == 3:
            occ_logits = occ_logits.unsqueeze(1)

        # CV prior + residual
        cv_traj = self._make_cv_traj(motion.to(traj_t.device), t_rel.to(traj_t.device))
        traj_res = traj_head
        traj_p = cv_traj + traj_res

        loss_occ = self._occ_loss(occ_logits, occ_t)
        loss_traj = self._traj_residual_loss(traj_res, traj_t, cv_traj, t_rel)
        loss = loss_occ + self.loss_cfg.traj_w * loss_traj

        # metrics
        ade_m, fde_m = ade_fde(traj_p, traj_t)
        ade_cv, fde_cv = ade_fde(cv_traj, traj_t)

        return loss, loss_occ, loss_traj, ade_m.mean(), fde_m.mean(), ade_cv.mean(), fde_cv.mean()

    def training_step(self, batch, batch_idx):
        loss, l_occ, l_traj, ade_m, fde_m, ade_cv, fde_cv = self._step(batch)
        self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log("train/occ", l_occ, prog_bar=False, on_step=True, on_epoch=True)
        self.log("train/traj", l_traj, prog_bar=False, on_step=True, on_epoch=True)
        self.log("train/ADE", ade_m, prog_bar=True, on_step=True, on_epoch=True)
        self.log("train/FDE", fde_m, prog_bar=True, on_step=True, on_epoch=True)
        self.log("train/CV_ADE", ade_cv, prog_bar=False, on_step=True, on_epoch=True)
        self.log("train/CV_FDE", fde_cv, prog_bar=False, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, l_occ, l_traj, ade_m, fde_m, ade_cv, fde_cv = self._step(batch)
        self.log("val/loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("val/occ", l_occ, prog_bar=False, on_step=False, on_epoch=True)
        self.log("val/traj", l_traj, prog_bar=False, on_step=False, on_epoch=True)
        self.log("val/ADE", ade_m, prog_bar=True, on_step=False, on_epoch=True)
        self.log("val/FDE", fde_m, prog_bar=True, on_step=False, on_epoch=True)
        self.log("val/CV_ADE", ade_cv, prog_bar=False, on_step=False, on_epoch=True)
        self.log("val/CV_FDE", fde_cv, prog_bar=False, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=50)
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sch, "interval": "epoch"}}

    def configure_gradient_clipping(self, optimizer, gradient_clip_val=None, gradient_clip_algorithm=None):
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=self.grad_clip)

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

class TemporalTransformer(nn.Module):
    def __init__(self, d: int = 384, nheads: int = 6, nlayers: int = 4, dropout: float = 0.1):
        super().__init__()
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d, nhead=nheads, dim_feedforward=4*d, dropout=dropout, batch_first=True, activation="gelu"
        )
        self.enc = nn.TransformerEncoder(enc_layer, num_layers=nlayers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, D]
        return self.enc(x)

class MultiViewVideoBackbone(nn.Module):
    """
    Input:  x [B, V, T, C, H, W]
    Output: z [B, D] pooled representation, plus per-token [B, V, T, D] if needed
    """
    def __init__(self, d: int = 384):
        super().__init__()
        # Lightweight CNN patch-embed (keeps it runnable on laptop)
        self.stem = nn.Sequential(
            nn.Conv2d(3, d//2, 7, stride=2, padding=3),
            nn.BatchNorm2d(d//2),
            nn.GELU(),
            nn.Conv2d(d//2, d, 3, stride=2, padding=1),
            nn.BatchNorm2d(d),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.temporal = TemporalTransformer(d=d, nheads=6, nlayers=4, dropout=0.1)
        self.view_fuse = nn.Sequential(
            nn.Linear(d, d),
            nn.GELU(),
            nn.Linear(d, d),
        )

    def forward(self, x: torch.Tensor):
        B, V, T, C, H, W = x.shape
        xt = rearrange(x, "b v t c h w -> (b v t) c h w")
        ft = self.stem(xt).flatten(1)  # [(B*V*T), D]
        ft = rearrange(ft, "(b v t) d -> b v t d", b=B, v=V, t=T)

        # temporal per view
        ft2 = rearrange(ft, "b v t d -> (b v) t d")
        ht = self.temporal(ft2)                        # [(B*V), T, D]
        hv = ht.mean(dim=1)                           # [(B*V), D]
        hv = rearrange(hv, "(b v) d -> b v d", b=B, v=V)

        # fuse views (mean + MLP)
        z = hv.mean(dim=1)                            # [B, D]
        z = self.view_fuse(z)                         # [B, D]
        return z, ft  # global + per-frame features

class BEVOccupancyHead(nn.Module):
    def __init__(self, d: int = 384, bev_h: int = 64, bev_w: int = 64):
        super().__init__()
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.mlp = nn.Sequential(
            nn.Linear(d, 2*d),
            nn.GELU(),
            nn.Linear(2*d, bev_h * bev_w),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        logits = self.mlp(z).view(z.size(0), 1, self.bev_h, self.bev_w)
        return logits

class TrajHead(nn.Module):
    def __init__(self, d: int = 384, horizon: int = 12):
        super().__init__()
        self.horizon = horizon
        self.mlp = nn.Sequential(
            nn.Linear(d, 2*d),
            nn.GELU(),
            nn.Linear(2*d, horizon * 2),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        xy = self.mlp(z).view(z.size(0), self.horizon, 2)
        return xy

class OpenDriveFM(nn.Module):
    """
    Multi-task:
      - BEV occupancy: logits [B,1,H,W]
      - future trajectory: xy [B,HORIZON,2]
    """
    def __init__(self, d: int = 384, bev_h: int = 64, bev_w: int = 64, horizon: int = 12):
        super().__init__()
        self.backbone = MultiViewVideoBackbone(d=d)
        self.occ = BEVOccupancyHead(d=d, bev_h=bev_h, bev_w=bev_w)
        self.traj = TrajHead(d=d, horizon=horizon)

    def forward(self, x: torch.Tensor):
        z, _ = self.backbone(x)
        occ_logits = self.occ(z)
        traj_xy = self.traj(z)
        return occ_logits, traj_xy

def occ_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    # target: [B,1,H,W] in {0,1}
    return F.binary_cross_entropy_with_logits(logits, target)

def traj_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    # pred/target: [B,H,2]
    return F.smooth_l1_loss(pred, target)

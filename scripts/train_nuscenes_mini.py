from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from opendrivefm.data.nuscenes_mini import NuScenesMiniMultiView
from opendrivefm.train.lightning_module import LitOpenDriveFM, LossCfg, _dl_kwargs


def read_rows(manifest: str) -> list[dict]:
    return [json.loads(l) for l in Path(manifest).read_text().splitlines() if l.strip()]


def scene_name(r: dict) -> str:
    return r.get("scene_name") or r.get("scene") or "unknown"


def split_by_scene(rows: list[dict], val_frac: float, seed: int) -> tuple[list[int], list[int], list[str]]:
    scenes = sorted({scene_name(r) for r in rows})
    rnd = random.Random(seed)
    rnd.shuffle(scenes)

    n_val = max(1, int(round(val_frac * len(scenes))))
    val_scenes = sorted(scenes[:n_val])
    val_set = set(val_scenes)

    train_idx, val_idx = [], []
    for i, r in enumerate(rows):
        (val_idx if scene_name(r) in val_set else train_idx).append(i)

    return train_idx, val_idx, val_scenes


@torch.no_grad()
def estimate_pos_frac(ds_subset: Subset) -> float:
    # exact scan (small dataset; fast enough)
    pos = 0.0
    tot = 0.0
    for i in range(len(ds_subset)):
        _, occ, _ = ds_subset[i]
        # occ: (1,H,W)
        pos += float(occ.sum().item())
        tot += float(occ.numel())
    return max(1e-8, pos / tot)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=str, default="artifacts/nuscenes_mini_manifest.jsonl")
    ap.add_argument("--label_root", type=str, default="artifacts/nuscenes_labels")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--val_frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--bev", type=int, default=64)
    ap.add_argument("--horizon", type=int, default=12)
    ap.add_argument("--occ_dice_w", type=float, default=0.7)
    ap.add_argument("--pos_weight_cap", type=float, default=10.0)
    ap.add_argument("--traj_beta", type=float, default=1.0)
    ap.add_argument("--clean_ckpt", action="store_true")
    args = ap.parse_args()

    pl.seed_everything(args.seed, workers=True)

    rows = read_rows(args.manifest)
    train_idx, val_idx, val_scenes = split_by_scene(rows, args.val_frac, args.seed)

    print(f"VAL_SCENES({len(val_scenes)}): {val_scenes}")

    ds = NuScenesMiniMultiView(
        args.manifest,
        image_size=(160, 90),
        frames=1,
        label_root=args.label_root,
    )

    train_ds = Subset(ds, train_idx)
    val_ds = Subset(ds, val_idx)

    print(f"TRAIN_ITEMS: {len(train_ds)} VAL_ITEMS: {len(val_ds)}")
    print("DL_KWARGS:", _dl_kwargs())

    pos_frac = estimate_pos_frac(train_ds)
    pos_weight = float((1.0 - pos_frac) / pos_frac)
    print(f"OCC_POS_FRAC≈{pos_frac:.6f}  POS_WEIGHT≈{pos_weight:.2f} (cap={args.pos_weight_cap})")

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, **_dl_kwargs())
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False, **_dl_kwargs())

    ckpt_dir = Path("artifacts") / "checkpoints_nuscenes"
    if args.clean_ckpt and ckpt_dir.exists():
        for p in ckpt_dir.glob("*.ckpt"):
            p.unlink()
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    ckpt = ModelCheckpoint(
        dirpath=str(ckpt_dir),
        save_last=True,
        save_top_k=1,
        monitor="val/loss",
        mode="min",
        filename="epoch={epoch:02d}-step={step}",
    )

    loss_cfg = LossCfg(
        occ_dice_w=args.occ_dice_w,
        pos_weight=pos_weight,
        pos_weight_cap=args.pos_weight_cap,
        traj_beta=args.traj_beta,
    )

    model = LitOpenDriveFM(
        lr=args.lr,
        bev=args.bev,
        horizon=args.horizon,
        loss=loss_cfg,
        grad_clip=1.0,
    )

    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator="auto",
        devices="auto",
        precision="32-true",
        callbacks=[ckpt],
        log_every_n_steps=10,
        enable_checkpointing=True,
    )

    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    print("BEST:", ckpt.best_model_path)
    print("LAST:", str(ckpt_dir / "last.ckpt"))


if __name__ == "__main__":
    main()

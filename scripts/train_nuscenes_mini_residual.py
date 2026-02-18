from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

from opendrivefm.data.nuscenes_mini import NuScenesMiniMultiView
from opendrivefm.train.lightning_module import LitOpenDriveFM, _dl_kwargs


def split_by_scene(rows, seed: int, val_frac: float):
    scenes = sorted({r["scene"] for r in rows})
    rng = random.Random(seed)
    rng.shuffle(scenes)
    n_val = max(1, int(round(len(scenes) * val_frac)))
    val_scenes = set(scenes[:n_val])
    idx_val = [i for i, r in enumerate(rows) if r["scene"] in val_scenes]
    idx_train = [i for i, r in enumerate(rows) if r["scene"] not in val_scenes]
    return idx_train, idx_val, sorted(val_scenes)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=str, default="artifacts/nuscenes_mini_manifest.jsonl")
    ap.add_argument("--label_root", type=str, default="artifacts/nuscenes_labels")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--val_frac_scenes", type=float, default=0.2)

    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--max_epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=3e-4)

    ap.add_argument("--image_hw_h", type=int, default=90)
    ap.add_argument("--image_hw_w", type=int, default=160)

    ap.add_argument("--ckpt_dir", type=str, default="artifacts/checkpoints_nuscenes_residual")
    args = ap.parse_args()

    rows = [json.loads(l) for l in Path(args.manifest).read_text().splitlines() if l.strip()]
    idx_train, idx_val, val_scenes = split_by_scene(rows, seed=args.seed, val_frac=args.val_frac_scenes)

    print("VAL_SCENES:", val_scenes)
    print("TRAIN_ITEMS:", len(idx_train), "VAL_ITEMS:", len(idx_val))

    ds = NuScenesMiniMultiView(
        args.manifest,
        image_hw=(args.image_hw_h, args.image_hw_w),
        frames=1,
        label_root=args.label_root,
        return_motion=True,
        return_trel=True,
    )

    dl_train = DataLoader(Subset(ds, idx_train), batch_size=args.batch_size, shuffle=True, **_dl_kwargs())
    dl_val = DataLoader(Subset(ds, idx_val), batch_size=args.batch_size, shuffle=False, **_dl_kwargs())

    m = LitOpenDriveFM(lr=args.lr)

    accel = "mps" if torch.backends.mps.is_available() else "cpu"
    ckpt_dir = Path(args.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    callbacks = [
        ModelCheckpoint(
            dirpath=str(ckpt_dir),
            filename="epoch{epoch:03d}-valFDE{val/FDE:.3f}",
            monitor="val/FDE",
            mode="min",
            save_last=True,
            save_top_k=2,
        ),
        LearningRateMonitor(logging_interval="epoch"),
    ]

    trainer = pl.Trainer(
        accelerator=accel,
        devices=1,
        max_epochs=args.max_epochs,
        precision="16-mixed" if accel == "mps" else "32-true",
        callbacks=callbacks,
        log_every_n_steps=10,
    )

    trainer.fit(m, train_dataloaders=dl_train, val_dataloaders=dl_val)
    print("DONE. Last checkpoint:", ckpt_dir / "last.ckpt")


if __name__ == "__main__":
    main()

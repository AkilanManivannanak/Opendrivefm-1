from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from opendrivefm.train.lightning_module import LitOpenDriveFM, make_synth_loaders


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--views", type=int, default=3)
    ap.add_argument("--frames", type=int, default=6)
    ap.add_argument("--bev", type=int, default=64)
    ap.add_argument("--horizon", type=int, default=12)
    ap.add_argument("--n_train", type=int, default=512)
    ap.add_argument("--n_val", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max_batches", type=int, default=50)
    args = ap.parse_args()

    _, val_loader = make_synth_loaders(
        batch_size=args.batch,
        n_train=args.n_train,
        n_val=args.n_val,
        seed=args.seed,
        views=args.views,
        frames=args.frames,
        bev=args.bev,
        horizon=args.horizon,
    )

    lit = LitOpenDriveFM.load_from_checkpoint(args.ckpt)
    lit.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu"))
    lit.to(device)

    losses, occ_losses, traj_losses = [], [], []

    for i, batch in enumerate(val_loader):
        if i >= args.max_batches:
            break
        x, occ_t, traj_t = batch
        x = x.to(device).float()
        occ_t = occ_t.to(device)
        traj_t = traj_t.to(device)

        occ_logits, traj_p = lit.model(x)
        l_occ = lit.hparams.w_occ * torch.tensor(1.0, device=device)  # placeholder weight usage
        l_traj = lit.hparams.w_traj * torch.tensor(1.0, device=device)

        # Use the same losses as training code
        from opendrivefm.models.model import occ_loss, traj_loss
        lo = occ_loss(occ_logits, occ_t)
        lt = traj_loss(traj_p, traj_t)
        loss = lit.hparams.w_occ * lo + lit.hparams.w_traj * lt

        losses.append(loss.item())
        occ_losses.append(lo.item())
        traj_losses.append(lt.item())

    metrics = {
        "ckpt": args.ckpt,
        "device": str(device),
        "val_loss_mean": float(sum(losses) / max(1, len(losses))),
        "val_occ_loss_mean": float(sum(occ_losses) / max(1, len(occ_losses))),
        "val_traj_loss_mean": float(sum(traj_losses) / max(1, len(traj_losses))),
        "batches": len(losses),
    }

    out = Path("artifacts") / "metrics.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))
    print("WROTE:", out)

if __name__ == "__main__":
    main()

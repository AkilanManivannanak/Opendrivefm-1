from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from opendrivefm.data.nuscenes_mini import NuScenesMiniMultiView
from opendrivefm.train.lightning_module import LitOpenDriveFM, _dl_kwargs


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


def _dilate_bool(m: torch.Tensor, r: int) -> torch.Tensor:
    if r <= 0:
        return m
    k = 2 * r + 1
    t = F.max_pool2d(m.float(), kernel_size=k, stride=1, padding=r)
    return t > 0.0


def _erode_bool(m: torch.Tensor, r: int) -> torch.Tensor:
    if r <= 0:
        return m
    return ~_dilate_bool(~m, r)


def postprocess(mask: torch.Tensor, open_r: int, erode_r: int) -> torch.Tensor:
    # mask: (B,1,H,W) bool
    # opening = erode then dilate
    if open_r > 0:
        mask = _dilate_bool(_erode_bool(mask, open_r), open_r)
    if erode_r > 0:
        mask = _erode_bool(mask, erode_r)
    return mask


def occ_metrics(pred: torch.Tensor, gt: torch.Tensor) -> dict[str, float]:
    # pred, gt bool tensors (H,W)
    tp = (pred & gt).sum().item()
    fp = (pred & ~gt).sum().item()
    fn = (~pred & gt).sum().item()
    union = tp + fp + fn
    iou = (tp / union) if union > 0 else 0.0
    dice = (2 * tp / (2 * tp + fp + fn)) if (2 * tp + fp + fn) > 0 else 0.0
    prec = (tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    rec = (tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    return {"iou": float(iou), "dice": float(dice), "precision": float(prec), "recall": float(rec)}


def traj_metrics(pred: torch.Tensor, gt: torch.Tensor) -> dict[str, float]:
    # pred, gt: (H,2)
    d = torch.linalg.norm(pred - gt, dim=-1)  # (H,)
    ade = d.mean().item()
    fde = d[-1].item()
    return {"ADE": float(ade), "FDE": float(fde)}


def save_viz(out_dir: Path, tok: str, gt01: np.ndarray, pr01: np.ndarray) -> None:
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)

    gt_path = out_dir / f"{tok}_occ_gt.png"
    pr_path = out_dir / f"{tok}_occ_pred.png"
    ov_path = out_dir / f"{tok}.png"

    plt.imsave(gt_path, gt01.astype(np.uint8), cmap="viridis")
    plt.imsave(pr_path, pr01.astype(np.uint8), cmap="viridis")

    # overlay: R=GT, G=PRED
    h, w = gt01.shape
    ov = np.zeros((h, w, 3), dtype=np.uint8)
    ov[..., 0] = (gt01 * 255).astype(np.uint8)
    ov[..., 1] = (pr01 * 255).astype(np.uint8)
    plt.imsave(ov_path, ov)


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--manifest", type=str, default="artifacts/nuscenes_mini_manifest.jsonl")
    ap.add_argument("--label_root", type=str, default="artifacts/nuscenes_labels")
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--val_frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--thr", type=float, default=0.3)
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--thr_min", type=float, default=0.05)
    ap.add_argument("--thr_max", type=float, default=0.95)
    ap.add_argument("--thr_steps", type=int, default=19)

    ap.add_argument("--pp_open", type=int, default=0, help="postprocess opening radius")
    ap.add_argument("--pp_erode", type=int, default=0, help="postprocess erosion radius")

    ap.add_argument("--viz", type=int, default=0)
    args = ap.parse_args()

    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")

    rows = read_rows(args.manifest)
    train_idx, val_idx, val_scenes = split_by_scene(rows, args.val_frac, args.seed)
    print(f"VAL_SCENES({len(val_scenes)}): {val_scenes}")

    ds = NuScenesMiniMultiView(args.manifest, image_size=(160, 90), frames=1, label_root=args.label_root)
    val_ds = Subset(ds, val_idx)

    print(f"VAL_ITEMS: {len(val_ds)}  (TRAIN_ITEMS would be {len(train_idx)})")
    print("DL_KWARGS:", _dl_kwargs())

    loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False, **_dl_kwargs())

    lit = LitOpenDriveFM.load_from_checkpoint(args.ckpt)
    lit.to(device)
    lit.eval()

    # collect per-item outputs for sweep
    probs_all = []
    gt_all = []
    traj_p_all = []
    traj_t_all = []
    toks = [rows[i]["sample_token"] for i in val_idx]

    losses = []
    loss_occ = []
    loss_traj = []

    for batch in loader:
        x, occ_t, traj_t = batch
        x = x.to(device)
        occ_t = occ_t.to(device)
        traj_t = traj_t.to(device)

        occ_logits, traj_p = lit.model(x)

        # losses
        l, lo, lt = lit._step((x, occ_t, traj_t))
        losses.append(float(l.item()))
        loss_occ.append(float(lo.item()))
        loss_traj.append(float(lt.item()))

        p = torch.sigmoid(occ_logits).detach().cpu()          # (B,1,H,W)
        g = (occ_t.detach().cpu() > 0.5)                      # bool
        probs_all.append(p)
        gt_all.append(g)
        traj_p_all.append(traj_p.detach().cpu())
        traj_t_all.append(traj_t.detach().cpu())

    probs = torch.cat(probs_all, dim=0)  # (N,1,H,W)
    gt = torch.cat(gt_all, dim=0)        # (N,1,H,W) bool
    traj_p = torch.cat(traj_p_all, dim=0)  # (N,H,2)
    traj_t = torch.cat(traj_t_all, dim=0)  # (N,H,2)

    # traj metrics (independent of threshold)
    ade_list = []
    fde_list = []
    for i in range(traj_p.size(0)):
        tm = traj_metrics(traj_p[i], traj_t[i])
        ade_list.append(tm["ADE"])
        fde_list.append(tm["FDE"])

    def run_thr(thr: float) -> tuple[dict[str, float], list[dict]]:
        pred = probs > thr  # bool
        pred = postprocess(pred, args.pp_open, args.pp_erode)

        occ_iou = []
        occ_dice = []
        occ_prec = []
        occ_rec = []

        items = []
        for i in range(pred.size(0)):
            m = occ_metrics(pred[i, 0], gt[i, 0])
            occ_iou.append(m["iou"])
            occ_dice.append(m["dice"])
            occ_prec.append(m["precision"])
            occ_rec.append(m["recall"])

            items.append(
                {
                    "sample_token": toks[i],
                    "thr": float(thr),
                    "occ": m,
                    "traj": {"ADE": float(ade_list[i]), "FDE": float(fde_list[i])},
                }
            )

        metrics = {
            "thr": float(thr),
            "occ_iou_mean": float(np.mean(occ_iou)),
            "occ_dice_mean": float(np.mean(occ_dice)),
            "occ_precision_mean": float(np.mean(occ_prec)),
            "occ_recall_mean": float(np.mean(occ_rec)),
            "occ_iou_median": float(np.median(occ_iou)),
            "occ_dice_median": float(np.median(occ_dice)),
        }
        return metrics, items, pred

    curve = []
    best = None
    best_thr = args.thr
    best_score = -1.0
    best_pred = None
    best_items = None

    thrs = [args.thr]
    if args.sweep:
        thrs = list(np.linspace(args.thr_min, args.thr_max, args.thr_steps))

    for thr in thrs:
        m, items, pred = run_thr(float(thr))
        curve.append(m)
        # pick best by IoU mean (you can change to dice if you want)
        if m["occ_iou_mean"] > best_score:
            best_score = m["occ_iou_mean"]
            best = m
            best_thr = float(thr)
            best_pred = pred
            best_items = items

    # write per-item JSONL (best threshold)
    items_path = Path("artifacts") / "nuscenes_eval_items.jsonl"
    with items_path.open("w") as f:
        for it in best_items:
            f.write(json.dumps(it) + "\n")

    # optional visualization
    viz_dir = Path("artifacts") / "nuscenes_eval"
    if args.viz and args.viz > 0:
        n = min(args.viz, best_pred.size(0))
        for i in range(n):
            tok = toks[i]
            gt01 = gt[i, 0].numpy().astype(np.uint8)
            pr01 = best_pred[i, 0].numpy().astype(np.uint8)
            save_viz(viz_dir, tok, gt01, pr01)
        print("VIZ_DIR:", viz_dir)

    out = Path("artifacts") / "nuscenes_eval_metrics.json"
    metrics = {
        "ckpt": args.ckpt,
        "device": device,
        "seed": args.seed,
        "val_frac_scenes": args.val_frac,
        "val_scenes": val_scenes,
        "val_items": int(len(val_ds)),
        "batches": int(len(losses)),
        "val_loss_mean": float(np.mean(losses)),
        "val_occ_loss_mean": float(np.mean(loss_occ)),
        "val_traj_loss_mean": float(np.mean(loss_traj)),
        "traj_ADE_mean": float(np.mean(ade_list)),
        "traj_FDE_mean": float(np.mean(fde_list)),
        "traj_ADE_median": float(np.median(ade_list)),
        "best_thr": float(best_thr),
        "best": best,
        "curve": curve if args.sweep else None,
        "postprocess": {"open_r": int(args.pp_open), "erode_r": int(args.pp_erode)},
    }
    out.write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))
    print("WROTE:", out)
    print("WROTE:", items_path)


if __name__ == "__main__":
    main()

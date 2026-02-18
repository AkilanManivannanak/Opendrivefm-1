from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from pyquaternion import Quaternion
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud


def _dilate_binary(mask01: np.ndarray, r: int) -> np.ndarray:
    if r <= 0:
        return mask01
    k = 2 * r + 1
    t = torch.from_numpy(mask01[None, None].astype(np.float32))
    t = F.max_pool2d(t, kernel_size=k, stride=1, padding=r)
    return (t[0, 0].numpy() > 0.0).astype(np.uint8)


def _lidar_sd(nusc: NuScenes, sample: dict) -> dict:
    return nusc.get("sample_data", sample["data"]["LIDAR_TOP"])


def _ego_pose_from_sd(nusc: NuScenes, sd: dict) -> tuple[np.ndarray, np.ndarray]:
    pose = nusc.get("ego_pose", sd["ego_pose_token"])
    R = Quaternion(pose["rotation"]).rotation_matrix.astype(np.float32)
    t = np.array(pose["translation"], dtype=np.float32)
    return R, t


def _timestamp_sec(sd: dict) -> float:
    return float(sd["timestamp"]) / 1e6


def build_occ(
    nusc: NuScenes,
    sample: dict,
    bev: int,
    extent_m: float,
    z_min: float,
    z_max: float,
    dilate_r: int,
) -> np.ndarray:
    sd = _lidar_sd(nusc, sample)
    cs = nusc.get("calibrated_sensor", sd["calibrated_sensor_token"])

    pc = LidarPointCloud.from_file(str(Path(nusc.dataroot) / sd["filename"]))

    R = Quaternion(cs["rotation"]).rotation_matrix
    t = np.array(cs["translation"])
    pc.rotate(R)
    pc.translate(t)

    pts = pc.points[:3, :]
    x, y, z = pts[0], pts[1], pts[2]

    keep = (z >= z_min) & (z <= z_max)
    x, y = x[keep], y[keep]

    keep2 = (x >= -extent_m) & (x <= extent_m) & (y >= -extent_m) & (y <= extent_m)
    x, y = x[keep2], y[keep2]

    occ = np.zeros((bev, bev), dtype=np.uint8)
    if x.size == 0:
        return occ[None].astype(np.float32)

    s = (2.0 * extent_m) / bev
    r = np.floor((extent_m - x) / s).astype(np.int32)
    c = np.floor((y + extent_m) / s).astype(np.int32)

    r = np.clip(r, 0, bev - 1)
    c = np.clip(c, 0, bev - 1)
    occ[r, c] = 1
    occ = _dilate_binary(occ, dilate_r)
    return occ[None].astype(np.float32)


def build_prev_motion_ego0(nusc: NuScenes, sample0: dict) -> tuple[float, np.ndarray]:
    """
    Estimate velocity from prev->current, expressed in CURRENT ego frame (ego0).
    Returns: (dt_prev_sec, vxy_prev_mps[2])
    """
    prev_tok = sample0.get("prev", "")
    if not prev_tok:
        return 0.0, np.zeros((2,), dtype=np.float32)

    s_prev = nusc.get("sample", prev_tok)

    sd0 = _lidar_sd(nusc, sample0)
    sdp = _lidar_sd(nusc, s_prev)

    t0 = _timestamp_sec(sd0)
    tp = _timestamp_sec(sdp)
    dt = t0 - tp
    if dt <= 1e-6:
        return 0.0, np.zeros((2,), dtype=np.float32)

    R0, x0 = _ego_pose_from_sd(nusc, sd0)   # CURRENT pose
    _,  xp = _ego_pose_from_sd(nusc, sdp)   # PREV translation (global)

    # displacement prev->current in CURRENT ego frame:
    disp_ego0 = (R0.T @ (x0 - xp)).astype(np.float32)  # (3,)
    vxy = (disp_ego0[:2] / float(dt)).astype(np.float32)
    return float(dt), vxy


def build_traj_ego_and_trel(nusc: NuScenes, sample0: dict, horizon: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Future ego positions expressed in current ego frame (ego0), plus time offsets.
    Returns:
      traj: (H,2) meters
      t_rel: (H,) seconds from current time
    """
    sd0 = _lidar_sd(nusc, sample0)
    R0, x0 = _ego_pose_from_sd(nusc, sd0)
    t0 = _timestamp_sec(sd0)

    traj = []
    t_rel = []
    s = sample0
    last_xy = np.zeros((2,), dtype=np.float32)
    last_tr = 0.0

    for _ in range(horizon):
        nxt = s.get("next", "")
        if not nxt:
            traj.append(last_xy.copy())
            t_rel.append(last_tr)
            continue

        s = nusc.get("sample", nxt)
        sdk = _lidar_sd(nusc, s)
        _, xk = _ego_pose_from_sd(nusc, sdk)
        tk = _timestamp_sec(sdk)

        p_rel = (R0.T @ (xk - x0)).astype(np.float32)
        xy = p_rel[:2]
        tr = float(tk - t0)

        traj.append(xy)
        t_rel.append(tr)

        last_xy = xy
        last_tr = tr

    return np.stack(traj, axis=0).astype(np.float32), np.array(t_rel, dtype=np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, required=True)
    ap.add_argument("--version", type=str, default="v1.0-mini")
    ap.add_argument("--manifest", type=str, required=True)
    ap.add_argument("--out_dir", type=str, default="artifacts/nuscenes_labels")

    ap.add_argument("--bev", type=int, default=64)
    ap.add_argument("--extent_m", type=float, default=20.0)
    ap.add_argument("--horizon", type=int, default=12)

    ap.add_argument("--z_min", type=float, default=-1.2)
    ap.add_argument("--z_max", type=float, default=3.0)
    ap.add_argument("--dilate", type=int, default=2)

    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    nusc = NuScenes(version=args.version, dataroot=args.data_root, verbose=False)
    manifest = Path(args.manifest)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = [json.loads(l) for l in manifest.read_text().splitlines() if l.strip()]
    if args.limit and args.limit > 0:
        rows = rows[: args.limit]

    for i, r in enumerate(rows):
        sample0 = nusc.get("sample", r["sample_token"])

        occ = build_occ(
            nusc,
            sample0,
            bev=args.bev,
            extent_m=args.extent_m,
            z_min=args.z_min,
            z_max=args.z_max,
            dilate_r=args.dilate,
        )

        traj, t_rel = build_traj_ego_and_trel(nusc, sample0, horizon=args.horizon)
        dt_prev, vxy_prev = build_prev_motion_ego0(nusc, sample0)

        np.savez_compressed(
            out_dir / f'{r["sample_token"]}.npz',
            occ=occ,
            traj=traj,
            t_rel=t_rel,
            vxy_prev=vxy_prev,
            dt_prev=np.float32(dt_prev),
        )

        if (i + 1) % 25 == 0 or (i + 1) == len(rows):
            print(f"labels: {i+1}/{len(rows)}")

    print("WROTE:", out_dir, "files=", len(rows))


if __name__ == "__main__":
    main()

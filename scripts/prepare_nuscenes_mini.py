from __future__ import annotations
import json
from pathlib import Path
from tqdm import tqdm

def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", type=str, default="data/nuscenes")
    p.add_argument("--version", type=str, default="v1.0-mini")
    p.add_argument("--out", type=str, default="artifacts/nuscenes_mini_manifest.jsonl")
    p.add_argument("--limit", type=int, default=0, help="0=all")
    args = p.parse_args()

    from nuscenes.nuscenes import NuScenes

    data_root = Path(args.data_root)
    nusc = NuScenes(version=args.version, dataroot=str(data_root), verbose=False)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    cams = [
        "CAM_FRONT",
        "CAM_FRONT_LEFT",
        "CAM_FRONT_RIGHT",
        "CAM_BACK",
        "CAM_BACK_LEFT",
        "CAM_BACK_RIGHT",
    ]

    rows = []
    for scene in tqdm(nusc.scene, desc="scenes"):
        sample_token = scene["first_sample_token"]
        while sample_token:
            sample = nusc.get("sample", sample_token)

            # multi-view image paths at this timestamp
            img_paths = {}
            ok = True
            for cam in cams:
                sd_token = sample["data"].get(cam)
                if sd_token is None:
                    ok = False
                    break
                sd = nusc.get("sample_data", sd_token)
                img_paths[cam] = str((data_root / sd["filename"]).as_posix())
            if not ok:
                sample_token = sample["next"]
                continue

            # Ego pose (can become traj target later)
            # For now we store tokens; label building can be a later step.
            rows.append({
                "scene": scene["name"],
                "sample_token": sample_token,
                "cams": img_paths,
            })

            sample_token = sample["next"]
            if args.limit and len(rows) >= args.limit:
                break
        if args.limit and len(rows) >= args.limit:
            break

    with out.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    print("WROTE:", out, "rows=", len(rows))

if __name__ == "__main__":
    main()

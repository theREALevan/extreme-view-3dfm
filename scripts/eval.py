#!/usr/bin/env python3
import argparse
import builtins
import os
import sys
from contextlib import contextmanager
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "models" / "vggt"))

import numpy as np
import torch
from tqdm import tqdm

from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images


@contextmanager
def _suppress_shape_warning():
    _print = builtins.print
    def _filter(*args, **kwargs):
        if args and "Found images with different shapes" in str(args[0]):
            return
        _print(*args, **kwargs)
    builtins.print = _filter
    try:
        yield
    finally:
        builtins.print = _print
from vggt.utils.pose_enc import pose_encoding_to_extri_intri

DATASET_CHOICES = {
    "unscene-t": "data/pairwise/UnScenePairs-t.npy",
    "unscene": "data/pairwise/UnScenePairs.npy",
}


def geodesic_angle_deg(m1, m2):
    batch = m1.shape[0]
    m = torch.bmm(m1.to(torch.float64), m2.to(torch.float64).transpose(1, 2))
    cos = (m[:, 0, 0] + m[:, 1, 1] + m[:, 2, 2] - 1) / 2
    cos = cos.clamp(-1.0, 1.0)
    return torch.acos(cos).float() * 180.0 / np.pi


def normalize_vector(v):
    v_mag = torch.sqrt(v.pow(2).sum(1)).clamp(min=1e-8)
    return v / v_mag.view(-1, 1).expand_as(v)


def quat_to_rot(qw, qx, qy, qz):
    q = normalize_vector(torch.stack([qw, qx, qy, qz], dim=1))
    qw, qx, qy, qz = q[:, 0:1], q[:, 1:2], q[:, 2:3], q[:, 3:4]
    xx, yy, zz = qx * qx, qy * qy, qz * qz
    xy, xz, yz = qx * qy, qx * qz, qy * qz
    xw, yw, zw = qx * qw, qy * qw, qz * qw
    row0 = torch.cat((1 - 2 * yy - 2 * zz, 2 * xy - 2 * zw, 2 * xz + 2 * yw), 1)
    row1 = torch.cat((2 * xy + 2 * zw, 1 - 2 * xx - 2 * zz, 2 * yz - 2 * xw), 1)
    row2 = torch.cat((2 * xz - 2 * yw, 2 * yz + 2 * xw, 1 - 2 * xx - 2 * yy), 1)
    return torch.stack([row0, row1, row2], dim=1)


def evaluate_pair(model, img1_path, img2_path, gt_R1, gt_R2, device, dtype):
    with _suppress_shape_warning():
        images = load_and_preprocess_images([img1_path, img2_path]).to(device)
    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=dtype):
            out = model.aggregator(images[None])
            pose_enc = model.camera_head(out[0])[-1]
            extrinsic, _ = pose_encoding_to_extri_intri(pose_enc, images.shape[-2:])
    pred_R1 = extrinsic[0, 0, :3, :3]
    pred_R2 = extrinsic[0, 1, :3, :3]
    pred_rel_R = pred_R2 @ pred_R1.T
    gt_rel_R = gt_R2 @ gt_R1.T
    rot_err = geodesic_angle_deg(pred_rel_R.unsqueeze(0), gt_rel_R.unsqueeze(0)).item()
    return rot_err


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", choices=list(DATASET_CHOICES), required=True)
    parser.add_argument("--base_dir", type=str, required=True)
    parser.add_argument("--ckpt", type=str, default="")
    parser.add_argument("--out", type=str, required=True)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if (device == "cuda" and torch.cuda.get_device_capability()[0] >= 8) else torch.float16
    data_path = REPO_ROOT / DATASET_CHOICES[args.data]
    base_dir = Path(args.base_dir)
    os.makedirs(args.out, exist_ok=True)

    model = VGGT.from_pretrained("facebook/VGGT-1B").to(device)
    if args.ckpt:
        ckpt_path = REPO_ROOT / args.ckpt
        ckpt = torch.load(ckpt_path, map_location=device)
        state = ckpt.get("bias_state_dict", ckpt.get("model"))
        if state is not None:
            state = {k.replace("module.", ""): v for k, v in state.items()}
            model.load_state_dict(state, strict=False)
    model.eval()

    test_data = np.load(data_path, allow_pickle=True).item()
    rot_by_overlap = {"large": [], "small": [], "none": []}

    for idx in tqdm(test_data.keys()):
        p = test_data[idx]
        overlap = p["overlap_amount"].lower()
        img1_path = base_dir / p["img1"]["path"]
        img2_path = base_dir / p["img2"]["path"]
        q1 = torch.tensor([[p["img1"]["qw"], p["img1"]["qx"], p["img1"]["qy"], p["img1"]["qz"]]], device=device)
        q2 = torch.tensor([[p["img2"]["qw"], p["img2"]["qx"], p["img2"]["qy"], p["img2"]["qz"]]], device=device)
        gt_R1 = quat_to_rot(q1[:, 0], q1[:, 1], q1[:, 2], q1[:, 3]).squeeze(0)
        gt_R2 = quat_to_rot(q2[:, 0], q2[:, 1], q2[:, 2], q2[:, 3]).squeeze(0)
        rot_err = evaluate_pair(model, str(img1_path), str(img2_path), gt_R1, gt_R2, device, dtype)
        rot_by_overlap[overlap].append(rot_err)

    results = {}
    for ov in ["large", "small", "none"]:
        if not rot_by_overlap[ov]:
            continue
        ro = rot_by_overlap[ov]
        results[ov] = {
            "rot_mge": float(np.median(ro)),
            "rot_rra15": float(np.mean(np.array(ro) <= 15)),
            "rot_rra30": float(np.mean(np.array(ro) <= 30)),
        }
        print(f"\n{ov}: MGE={results[ov]['rot_mge']:.2f}° RRA15={results[ov]['rot_rra15']*100:.1f}% RRA30={results[ov]['rot_rra30']*100:.1f}%")

    stem = args.data
    txt_path = Path(args.out) / f"eval_{stem}.txt"
    with open(txt_path, "w") as f:
        f.write(f"data={data_path}\nckpt={args.ckpt or 'base'}\nbase_dir={base_dir}\n\n")
        f.write(f"{'Overlap':10} {'MGE':>8} {'RRA15':>8} {'RRA30':>8}\n")
        for ov in ["large", "small", "none"]:
            if ov in results:
                r = results[ov]
                f.write(f"{ov:10} {r['rot_mge']:8.2f} {r['rot_rra15']*100:7.1f}% {r['rot_rra30']*100:7.1f}%\n")
    np.save(Path(args.out) / f"eval_{stem}.npy", {"rot_by_overlap": rot_by_overlap, "results": results})
    print(f"Wrote {txt_path} and eval_{stem}.npy")


if __name__ == "__main__":
    main()

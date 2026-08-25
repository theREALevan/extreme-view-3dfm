#!/usr/bin/env python3
import argparse
import builtins
import math
import os
import sys
from contextlib import contextmanager
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "models" / "vggt"))
sys.path.insert(0, str(REPO_ROOT / "models" / "pi3"))
sys.path.insert(0, str(REPO_ROOT / "models" / "worldmirror"))

import numpy as np
import torch
import torch.distributed as dist
from PIL import Image
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, Sampler
from torchvision import transforms
from tqdm import tqdm

from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri

_to_tensor = transforms.ToTensor()

# Per model: transformer blocks whose biases are tuned, whether the first camera is
# anchored to identity (1_a in the paper), and the layer selection used in the paper.
SPECS = {
    "vggt": {
        "blocks": ("aggregator.frame_blocks", "aggregator.global_blocks"),
        "anchor": True,
        "layers": "4,11,17,23",
    },
    "pi3": {
        "blocks": ("decoder",),
        "anchor": False,
        "layers": "8,24,26,27,28,29,30,31,32",
    },
    "wm": {
        "blocks": ("visual_geometry_transformer.frame_blocks", "visual_geometry_transformer.global_blocks"),
        "anchor": True,
        "layers": "4,11,17,23",
    },
}
BIASES = ("attn.qkv.bias", "attn.proj.bias", "mlp.fc1.bias", "mlp.fc2.bias")


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


def parse_layers(spec, total):
    """'-1' for all, '2-12' for a range, '4,11,17,23' for specific layers."""
    if spec == "-1":
        return list(range(total))
    idx = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = map(int, part.split("-"))
            idx.extend(range(lo, hi + 1))
        else:
            idx.append(int(part))
    return sorted({i for i in idx if 0 <= i < total})


def freeze_all_but_bias(model, spec, layers):
    """Freeze everything, then unfreeze qkv/proj/fc1/fc2 biases in the selected blocks."""
    for p in model.parameters():
        p.requires_grad = False
    names = []
    for attr in spec["blocks"]:
        blocks = model
        for part in attr.split("."):
            blocks = getattr(blocks, part)
        for i in parse_layers(layers, len(blocks)):
            for name, param in blocks[i].named_parameters():
                if name in BIASES:
                    param.requires_grad = True
                    names.append(f"{attr}.{i}.{name}")
    return names


def normalize(v):
    return v / v.norm(dim=-1, keepdim=True).clamp(min=1e-8)


def quat_to_rotmat(quat):
    """(..., 4) unit quaternion in (w,x,y,z) -> (..., 3, 3) rotation matrix."""
    w, x, y, z = normalize(quat).unbind(-1)
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    xw, yw, zw = x * w, y * w, z * w
    row0 = torch.stack([1 - 2 * (yy + zz), 2 * (xy - zw), 2 * (xz + yw)], dim=-1)
    row1 = torch.stack([2 * (xy + zw), 1 - 2 * (xx + zz), 2 * (yz - xw)], dim=-1)
    row2 = torch.stack([2 * (xz - yw), 2 * (yz + xw), 1 - 2 * (xx + yy)], dim=-1)
    return torch.stack([row0, row1, row2], dim=-2)


def geodesic_loss(R1, R2):
    """Mean SO(3) geodesic distance in radians."""
    m = torch.bmm(R1.to(torch.float64), R2.to(torch.float64).transpose(1, 2))
    cos = ((m[:, 0, 0] + m[:, 1, 1] + m[:, 2, 2] - 1.0) * 0.5).clamp(-1.0 + 1e-7, 1.0 - 1e-7)
    return torch.acos(cos).mean().float()


def rotation_loss(R_pred, R1_gt, R2_gt, anchor):
    """L = L_geo(R_rel_pred, R_rel_gt) + 1_a * L_geo(R1_pred, I)."""
    rel_pred = torch.bmm(R_pred[:, 1], R_pred[:, 0].transpose(1, 2))
    rel_gt = torch.bmm(R2_gt, R1_gt.transpose(1, 2))
    loss = geodesic_loss(rel_pred, rel_gt)
    if anchor:
        eye = torch.eye(3, device=R_pred.device).expand(R_pred.shape[0], 3, 3)
        loss = loss + geodesic_loss(R_pred[:, 0], eye)
    return loss


class PairDataset(Dataset):
    """Image pairs stored as a dict of {img1, img2} entries; each image carries a path and a
    world-to-camera quaternion (qw, qx, qy, qz). See the README for how to build one."""

    def __init__(self, npy_file, base_dir=""):
        self.data = np.load(npy_file, allow_pickle=True).item()
        self.keys = list(self.data.keys())
        self.base_dir = Path(base_dir) if base_dir else None

    def __len__(self):
        return len(self.keys)

    def _build(self, info):
        path = info["path"]
        if self.base_dir is not None and not os.path.isabs(path):
            path = str(self.base_dir / path)
        quat = torch.tensor([info["qw"], info["qx"], info["qy"], info["qz"]], dtype=torch.float32)
        return path, quat

    def __getitem__(self, idx):
        item = self.data[self.keys[idx]]
        p1, q1 = self._build(item["img1"])
        p2, q2 = self._build(item["img2"])
        return p1, p2, q1, q2


def load_images_pi3_wm(paths, pixel_limit=255000):
    """Resize every image to one common size: a multiple of 14, under pixel_limit,
    with the aspect ratio of the first image."""
    images = [Image.open(p).convert("RGB") for p in paths]
    W, H = images[0].size
    scale = math.sqrt(pixel_limit / (W * H)) if W * H > 0 else 1
    W_target, H_target = W * scale, H * scale
    k, m = round(W_target / 14), round(H_target / 14)
    while (k * 14) * (m * 14) > pixel_limit:
        if k / m > W_target / H_target:
            k -= 1
        else:
            m -= 1
    size = (max(1, k) * 14, max(1, m) * 14)
    return torch.stack([_to_tensor(im.resize(size, Image.Resampling.LANCZOS)) for im in images])


def make_collate(model_name):
    def collate(batch):
        paths1, paths2, q1, q2 = zip(*batch)
        try:
            # Load all 2B images at one size, then regroup as (B, 2, 3, H, W).
            with _suppress_shape_warning():
                paths = list(paths1) + list(paths2)
                imgs = load_and_preprocess_images(paths) if model_name == "vggt" else load_images_pi3_wm(paths)
            imgs = imgs.view(2, len(batch), *imgs.shape[1:]).permute(1, 0, 2, 3, 4)
        except Exception as e:  # truncated or unreadable image
            print(f"\nSkipping batch: {e}")
            return None
        return imgs.contiguous(), torch.stack(q1), torch.stack(q2)

    return collate


class ReplacementSampler(Sampler):
    """Uniform with-replacement sampling, sharded across ranks."""

    def __init__(self, n, rank, world, seed=0):
        self.n, self.rank, self.world, self.seed, self.epoch = n, rank, world, seed, 0
        self.num_samples = n // world

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        idx = torch.randint(0, self.n, (self.num_samples * self.world,), generator=g).tolist()
        return iter(idx[self.rank::self.world])

    def __len__(self):
        return self.num_samples


def build_model(name, device):
    """Imported lazily so only the selected model's dependencies are needed."""
    if name == "vggt":
        from vggt.models.vggt import VGGT
        return VGGT.from_pretrained("facebook/VGGT-1B").to(device)
    if name == "pi3":
        from pi3.models.pi3 import Pi3
        return Pi3.from_pretrained("yyfz233/Pi3").to(device)
    from src.models.models.worldmirror import WorldMirror
    model = WorldMirror.from_pretrained("tencent/HunyuanWorld-Mirror").to(device)
    if hasattr(model, "enable_cond"):
        model.enable_cond = False
    return model


class RotationNet(torch.nn.Module):
    """Wraps a backbone so that one forward call returns world-to-camera rotations."""

    def __init__(self, model, name):
        super().__init__()
        self.model, self.name = model, name

    def forward(self, images):
        if self.name == "vggt":
            pose_enc = self.model.camera_head(self.model.aggregator(images)[0])[-1]
            extrinsic, _ = pose_encoding_to_extri_intri(pose_enc, images.shape[-2:])
            return extrinsic[:, :, :3, :3]
        if self.name == "pi3":
            poses = self.model(images)["camera_poses"]
        else:
            poses = self.model(views={"img": images}, cond_flags=[0, 0, 0])["camera_poses"]
        return poses[:, :, :3, :3].transpose(-2, -1)  # camera-to-world -> world-to-camera


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=("vggt", "pi3", "wm"), default="vggt")
    parser.add_argument("--train_npy", type=str, required=True)
    parser.add_argument("--base_dir", type=str, default="")
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--layers", type=str, default="", help="Default: the layers used in the paper.")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=452)
    parser.add_argument("--resume", type=str, default="")
    args = parser.parse_args()

    rank = int(os.environ.get("LOCAL_RANK", 0))
    world = int(os.environ.get("WORLD_SIZE", 1))
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(rank)
    if world > 1:
        dist.init_process_group("nccl")
    torch.manual_seed(args.seed)
    dtype = torch.bfloat16 if (device.type == "cuda" and torch.cuda.get_device_capability()[0] >= 8) else torch.float16

    spec = SPECS[args.model]
    layers = args.layers or spec["layers"]
    out_dir = Path(args.out)

    dataset = PairDataset(args.train_npy, args.base_dir)
    sampler = ReplacementSampler(len(dataset), rank, world, seed=args.seed)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        collate_fn=make_collate(args.model),
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
    )

    model = build_model(args.model, device)
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt.get("bias_state_dict", ckpt.get("model")), strict=False)
    trainable = freeze_all_but_bias(model, spec, layers)
    if rank == 0:
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"{args.model}: layers {layers}, {len(trainable)} bias tensors, {n_params:,} trainable parameters")

    net = RotationNet(model, args.model)
    forward = DDP(net, device_ids=[rank], find_unused_parameters=True) if world > 1 else net
    forward.train()

    # Constant learning rate; only the biases above receive gradients.
    optimizer = torch.optim.AdamW([p for p in net.parameters() if p.requires_grad], lr=args.lr, weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(1, args.epochs + 1):
        sampler.set_epoch(epoch)
        bar = tqdm(loader, desc=f"Epoch {epoch}/{args.epochs}", file=sys.stdout) if rank == 0 else loader
        total, count = 0.0, 0
        for batch in bar:
            if batch is None:
                continue
            images, q1, q2 = (t.to(device, non_blocking=True) for t in batch)
            with torch.cuda.amp.autocast(dtype=dtype):
                R_pred = forward(images)
            loss = rotation_loss(R_pred.float(), quat_to_rotmat(q1), quat_to_rotmat(q2), spec["anchor"])

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_([p for p in net.parameters() if p.grad is not None], 1.0)
            if not torch.isfinite(grad_norm):
                optimizer.zero_grad(set_to_none=True)
                scaler.update()
                continue
            scaler.step(optimizer)
            scaler.update()

            total, count = total + loss.item(), count + 1
            if rank == 0:
                bar.set_postfix(loss=f"{total / count:.4f}")

        if rank == 0:
            mean_loss = total / max(count, 1)
            out_dir.mkdir(parents=True, exist_ok=True)
            ckpt_path = out_dir / f"{args.model}_bias_epoch_{epoch}.pth"
            torch.save({
                "bias_state_dict": {n: p.detach().cpu().clone() for n, p in model.named_parameters() if p.requires_grad},
                "epoch": epoch,
                "loss": mean_loss,
                "config": {"model": args.model, "layers": layers, "lr": args.lr,
                           "batch_size": args.batch_size, "world_size": world},
            }, ckpt_path)
            print(f"\nEpoch {epoch}: loss={mean_loss:.4f}, wrote {ckpt_path}")
        if world > 1:
            dist.barrier()

    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()

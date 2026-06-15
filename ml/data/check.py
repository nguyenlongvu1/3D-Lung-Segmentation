"""Sanity-check pipeline dữ liệu trước khi train.

Load 1 mẫu qua transforms, in shape/dtype/khoảng giá trị, và lưu ảnh overlay
1 lát cắt để kiểm tra bằng mắt. Chạy SAU khi đã tải data (Bước 2):

    python -m ml.data.check
    python -m ml.data.check --config path.yaml --out results/check.png
"""
import argparse
import os

import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")  # backend không cần màn hình
import matplotlib.pyplot as plt
from monai.apps import DecathlonDataset

from ml.config import load_config
from ml.data.transforms import get_transforms


def _describe(name, t):
    uniq = torch.unique(t)
    head = uniq[:8].tolist()
    print(
        f"  {name:12s} shape={tuple(t.shape)} dtype={t.dtype} "
        f"min={float(t.min()):.3f} max={float(t.max()):.3f} "
        f"n_unique={uniq.numel()} unique[:8]={[round(x, 2) for x in head]}"
    )


def _make_ds(cfg, section, train):
    return DecathlonDataset(
        root_dir=cfg.data.root,
        task=cfg.data.task,
        section=section,
        transform=get_transforms(cfg, train=train),
        download=False,
        val_frac=cfg.data.val_frac,
        cache_rate=0.0,      # không cache để chạy nhanh, nhẹ RAM
        num_workers=0,       # tránh rắc rối multiprocessing trên Windows
    )


def main():
    parser = argparse.ArgumentParser(description="Sanity-check pipeline dữ liệu")
    parser.add_argument("--config", default=None)
    parser.add_argument("--out", default="results/sanity_check.png")
    args = parser.parse_args()
    cfg = load_config(args.config)

    # 1) Volume validation (toàn bộ, sau tiền xử lý — KHÔNG augment)
    val_ds = _make_ds(cfg, "validation", train=False)
    print(f"Số mẫu validation: {len(val_ds)}")
    sample = val_ds[0]
    img, lbl = sample["image"], sample["label"]
    print("\n--- Volume validation (kỳ vọng: image∈[0,1], label∈{0,1}) ---")
    _describe("image", img)
    _describe("label", lbl)

    # 2) Một patch training (RandCropByPosNegLabeld trả về list các patch)
    train_ds = _make_ds(cfg, "training", train=True)
    patches = train_ds[0]
    patch = patches[0] if isinstance(patches, list) else patches
    print(f"\n--- Patch training (kỳ vọng shape [1, {', '.join(map(str, cfg.data.roi_size))}]) ---")
    _describe("patch img", patch["image"])
    _describe("patch lbl", patch["label"])

    # 3) Lưu ảnh overlay 1 lát cắt (chọn lát có nhiều nhãn nhất để thấy khối u)
    img_np, lbl_np = img[0].cpu().numpy(), lbl[0].cpu().numpy()
    if lbl_np.sum() > 0:
        z = int(lbl_np.sum(axis=(0, 1)).argmax())
    else:
        z = img_np.shape[2] // 2
        print("  (Lưu ý: volume này không có nhãn dương — lấy lát giữa)")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    plt.figure(figsize=(6, 6))
    plt.imshow(img_np[:, :, z], cmap="gray")
    overlay = np.ma.masked_where(lbl_np[:, :, z] == 0, lbl_np[:, :, z])
    plt.imshow(overlay, cmap="autumn", alpha=0.5)
    plt.title(f"Lát z={z} — ảnh CT + overlay nhãn")
    plt.axis("off")
    plt.savefig(args.out, bbox_inches="tight", dpi=120)
    plt.close()
    print(f"\n✅ Đã lưu ảnh kiểm tra: {args.out}")
    print("Mở ảnh đó xem nhãn (vùng đỏ) có nằm đúng trong phổi không.")


if __name__ == "__main__":
    main()

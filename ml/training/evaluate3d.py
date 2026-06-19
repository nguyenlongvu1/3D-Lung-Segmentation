"""Đánh giá model trên tập TEST (held-out) và ghi một dòng vào bảng benchmark.

Dùng đúng split cố định theo seed (model chưa từng thấy tập test).

    python -m ml.training.evaluate3d                      # model trong config
    python -m ml.training.evaluate3d --model swin_unetr   # đổi kiến trúc
"""
import argparse
import csv
import os
import time

import torch
from monai.inferers import sliding_window_inference

from ml.config import load_config
from ml.data.datamodule import get_test_loader
from ml.models.factory import build_model, count_parameters
from ml.training.metrics3d import SegMetrics


def _append_csv(path, row):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    exists = os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)
    print(f"Đã ghi benchmark -> {path}")


@torch.no_grad()
def evaluate(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = bool(cfg.train.amp) and device.type == "cuda"

    # Nạp checkpoint best của kiến trúc đang chọn
    ckpt = os.path.join(cfg.train.ckpt_dir, f"best_{cfg.model.name}.pth")
    if not os.path.exists(ckpt):
        raise FileNotFoundError(f"Chưa có checkpoint: {ckpt}. Hãy train trước (train3d).")
    model = build_model(cfg).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()

    test_loader = get_test_loader(cfg)
    metrics = SegMetrics(num_classes=cfg.model.out_channels)
    metrics.reset()

    roi = tuple(cfg.data.roi_size)
    infer_times = []
    for batch in test_loader:
        images = batch["image"].to(device)
        labels = batch["label"].to(device)
        t0 = time.time()
        with torch.amp.autocast("cuda", enabled=amp):
            outputs = sliding_window_inference(
                images, roi, cfg.train.sw_batch_size, model, overlap=cfg.train.sw_overlap
            )
        if device.type == "cuda":
            torch.cuda.synchronize()
        infer_times.append(time.time() - t0)
        metrics.update(outputs, labels)

    res = metrics.aggregate()
    row = {
        "model": cfg.model.name,
        "params_M": round(count_parameters(model) / 1e6, 2),
        "test_mean_dice": round(res["mean_dice"], 4),
        "test_mean_hd95": round(res["mean_hd95"], 2),
        "infer_s_per_vol": round(sum(infer_times) / len(infer_times), 2),
        "n_test": len(test_loader),
    }
    print("\n=== KẾT QUẢ TEST ===")
    for k, v in row.items():
        print(f"  {k}: {v}")
    _append_csv(os.path.join("results", "benchmark.csv"), row)
    return row


def main():
    p = argparse.ArgumentParser(description="Đánh giá test + ghi benchmark")
    p.add_argument("--config", default=None)
    p.add_argument("--model", default=None, help="ghi đè model.name")
    args = p.parse_args()
    cfg = load_config(args.config)
    if args.model:
        cfg.model.name = args.model
    evaluate(cfg)


if __name__ == "__main__":
    main()

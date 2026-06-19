"""Train loop 3D cho lung tumor segmentation (MONAI).

Ráp tất cả các mảnh: config + data (get_loaders) + model (build_model) +
metrics (SegMetrics). Dùng DiceCELoss, mixed-precision (AMP), validation bằng
sliding-window, early stopping, lưu best checkpoint + snapshot config.

Chạy:
    python -m ml.training.train3d                       # dùng config mặc định
    python -m ml.training.train3d --model swin_unetr    # đổi kiến trúc
    python -m ml.training.train3d --epochs 2            # smoke test nhanh
"""
import argparse
import os
import time

import torch
from torch.amp import autocast, GradScaler

from monai.losses import DiceCELoss
from monai.inferers import sliding_window_inference
from monai.utils import set_determinism

from ml.config import load_config, save_config
from ml.data.datamodule import get_loaders
from ml.models.factory import build_model, count_parameters
from ml.training.metrics3d import SegMetrics


@torch.no_grad()
def validate(model, val_loader, metrics, cfg, device, amp):
    """Chạy validation trên nguyên volume bằng sliding-window inference."""
    model.eval()
    metrics.reset()
    roi = tuple(cfg.data.roi_size)
    for batch in val_loader:
        images = batch["image"].to(device)
        labels = batch["label"].to(device)
        with autocast("cuda", enabled=amp):
            outputs = sliding_window_inference(
                images, roi, cfg.train.sw_batch_size, model, overlap=cfg.train.sw_overlap
            )
        metrics.update(outputs, labels)
    return metrics.aggregate()


def train(cfg):
    set_determinism(seed=cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = bool(cfg.train.amp) and device.type == "cuda"
    print(f"Device: {device} | Model: {cfg.model.name} | AMP: {amp}")

    train_loader, val_loader = get_loaders(cfg)
    model = build_model(cfg).to(device)
    print(f"Số tham số train được: {count_parameters(model):,}")

    # Loss: kết hợp Dice + CrossEntropy. to_onehot_y/softmax xử lý label [B,1,..] & logits.
    loss_fn = DiceCELoss(to_onehot_y=True, softmax=True)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay
    )
    scaler = GradScaler("cuda", enabled=amp)
    metrics = SegMetrics(num_classes=cfg.model.out_channels)

    # Tracking W&B (tùy chọn)
    use_wandb = bool(getattr(cfg.logging, "wandb", False))
    if use_wandb:
        import wandb

        wandb.init(
            project=cfg.logging.project,
            name=getattr(cfg.logging, "run_name", None),
            config={"model": cfg.model.name, "lr": cfg.train.lr, "roi": cfg.data.roi_size},
        )

    # Chuẩn bị thư mục lưu + snapshot config (tái lập)
    os.makedirs(cfg.train.ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(cfg.train.ckpt_dir, f"best_{cfg.model.name}.pth")
    save_config(cfg, os.path.join(cfg.train.ckpt_dir, f"config_{cfg.model.name}.yaml"))

    best_dice = -1.0
    patience = int(getattr(cfg.train, "early_stop_patience", 15))  # đơn vị: số lần validate
    counter = 0

    for epoch in range(cfg.train.max_epochs):
        model.train()
        epoch_loss = 0.0
        t0 = time.time()
        for batch in train_loader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            optimizer.zero_grad(set_to_none=True)
            with autocast("cuda", enabled=amp):
                outputs = model(images)
                loss = loss_fn(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += loss.item()
        epoch_loss /= max(1, len(train_loader))
        log = {"epoch": epoch + 1, "train_loss": epoch_loss}

        # ---- Validation định kỳ ----
        if (epoch + 1) % cfg.train.val_interval == 0:
            res = validate(model, val_loader, metrics, cfg, device, amp)
            mean_dice = res["mean_dice"]
            log.update({"val_dice": mean_dice, "val_hd95": res["mean_hd95"]})
            print(
                f"Epoch {epoch+1}/{cfg.train.max_epochs} | loss {epoch_loss:.4f} | "
                f"dice {mean_dice:.4f} | hd95 {res['mean_hd95']:.2f} | {time.time()-t0:.0f}s"
            )

            if mean_dice > best_dice:
                best_dice = mean_dice
                torch.save(model.state_dict(), ckpt_path)
                counter = 0
                print(f"  ↑ best dice {best_dice:.4f} -> đã lưu {ckpt_path}")
            else:
                counter += 1
                if counter >= patience:
                    print(f"Early stopping: không cải thiện sau {patience} lần validate.")
                    break
        else:
            print(f"Epoch {epoch+1}/{cfg.train.max_epochs} | loss {epoch_loss:.4f} | {time.time()-t0:.0f}s")

        if use_wandb:
            wandb.log(log)

    print(f"\nXong. Best val mean-dice: {best_dice:.4f} | checkpoint: {ckpt_path}")
    if use_wandb:
        wandb.finish()


def main():
    p = argparse.ArgumentParser(description="Train 3D segmentation (MONAI)")
    p.add_argument("--config", default=None)
    p.add_argument("--model", default=None, help="ghi đè model.name (monai_unet/swin_unetr)")
    p.add_argument("--epochs", type=int, default=None, help="ghi đè max_epochs (smoke test)")
    args = p.parse_args()

    cfg = load_config(args.config)
    if args.model:
        cfg.model.name = args.model
    if args.epochs:
        cfg.train.max_epochs = args.epochs
    train(cfg)


if __name__ == "__main__":
    main()

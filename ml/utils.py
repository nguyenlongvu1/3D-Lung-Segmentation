"""Tiện ích dùng chung cho pipeline train/eval/inference/deploy.

Gom các đoạn LẶP LẠI ở nhiều nơi thành một chỗ duy nhất:
  - chọn device (cuda nếu có),
  - bật AMP theo config + loại device,
  - đường dẫn checkpoint best của kiến trúc đang chọn (+ kiểm tra tồn tại),
  - nạp model đã train (build_model + load_state_dict + eval),
  - sliding-window inference theo tham số trong config,
  - chọn lát cắt có nhiều foreground nhất để hiển thị.
"""
import os

import torch
from monai.inferers import sliding_window_inference

from ml.models.factory import build_model


def get_device(device=None) -> torch.device:
    """Trả về device: ưu tiên tham số truyền vào, mặc định cuda nếu khả dụng."""
    if device is not None:
        return device
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def amp_enabled(cfg, device: torch.device) -> bool:
    """AMP chỉ bật khi config yêu cầu VÀ đang chạy trên CUDA."""
    return bool(cfg.train.amp) and device.type == "cuda"


def checkpoint_path(cfg) -> str:
    """Đường dẫn checkpoint best của kiến trúc đang chọn."""
    return os.path.join(cfg.train.ckpt_dir, f"best_{cfg.model.name}.pth")


def require_checkpoint(cfg) -> str:
    """Như checkpoint_path nhưng báo lỗi rõ ràng nếu chưa train."""
    ckpt = checkpoint_path(cfg)
    if not os.path.exists(ckpt):
        raise FileNotFoundError(f"Chưa có checkpoint: {ckpt}. Hãy train trước (train3d).")
    return ckpt


def load_trained_model(cfg, ckpt_path: str, device=None) -> torch.nn.Module:
    """Build model theo config, nạp trọng số từ checkpoint, chuyển sang eval."""
    device = get_device(device)
    model = build_model(cfg).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()
    return model


def sliding_window_predict(model, images, cfg):
    """Sliding-window inference dùng roi_size/sw_batch_size/sw_overlap trong config."""
    return sliding_window_inference(
        images,
        tuple(cfg.data.roi_size),
        cfg.train.sw_batch_size,
        model,
        overlap=cfg.train.sw_overlap,
    )


def best_foreground_slice(mask, axis: int = 2) -> int:
    """Chỉ số lát cắt (theo `axis`) có nhiều foreground nhất; rỗng -> lát giữa."""
    if mask.sum() > 0:
        reduce_axes = tuple(i for i in range(mask.ndim) if i != axis)
        return int(mask.sum(axis=reduce_axes).argmax())
    return mask.shape[axis] // 2

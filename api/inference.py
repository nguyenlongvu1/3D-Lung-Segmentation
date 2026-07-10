"""Inference 3D cho demo: nhận file NIfTI -> mask khối u + thể tích (ml).

Tái dùng transforms/kiến trúc của pipeline train (val transforms, không augment),
chạy sliding-window inference giống lúc validate để kết quả nhất quán.
"""
import numpy as np
import torch
from monai.transforms import (
    Compose,
    LoadImage,
    Orientation,
    Spacing,
    ScaleIntensityRange,
    EnsureType,
)

from ml.config import load_config
from ml.utils import get_device, load_trained_model, sliding_window_predict


def get_infer_transforms(cfg):
    """Tiền xử lý 1 ảnh CT (không nhãn) — giống val nhưng bỏ crop để giữ nguyên thể tích."""
    return Compose(
        [
            LoadImage(image_only=True, ensure_channel_first=True),  # -> [1, H, W, D]
            Orientation(axcodes="RAS"),
            Spacing(pixdim=tuple(cfg.data.spacing), mode="bilinear"),
            ScaleIntensityRange(
                a_min=cfg.data.a_min, a_max=cfg.data.a_max, b_min=0.0, b_max=1.0, clip=True
            ),
            EnsureType(),
        ]
    )


class LungTumorInference:
    def __init__(self, ckpt_path, cfg=None, device=None):
        self.cfg = cfg or load_config()
        self.device = get_device(device)
        self.model = load_trained_model(self.cfg, ckpt_path, self.device)
        self.tf = get_infer_transforms(self.cfg)
        self.voxel_mm3 = float(np.prod(self.cfg.data.spacing))

    @torch.no_grad()
    def predict(self, nifti_path):
        img = self.tf(nifti_path)                      # [1, H, W, D], giá trị [0,1]
        x = img.unsqueeze(0).to(self.device)           # [1, 1, H, W, D]
        logits = sliding_window_predict(self.model, x, self.cfg)
        mask = torch.argmax(logits, dim=1)[0].cpu().numpy().astype(np.uint8)  # [H, W, D]
        image = np.asarray(img[0].cpu())               # [H, W, D]
        tumor_ml = float(mask.sum()) * self.voxel_mm3 / 1000.0
        return {"image": image, "mask": mask, "tumor_ml": tumor_ml}

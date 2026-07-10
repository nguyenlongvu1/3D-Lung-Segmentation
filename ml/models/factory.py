"""Factory tạo model theo tên trong config — dùng chung cho train/eval/benchmark.

Hỗ trợ:
  - monai_unet : U-Net 3D của MONAI (baseline mạnh, ổn định).
  - swin_unetr : SwinUNETR — transformer encoder (điểm nhấn kiến trúc hiện đại).
"""
import inspect

import torch.nn as nn
from monai.networks.nets import UNet, SwinUNETR


def build_model(cfg) -> nn.Module:
    name = cfg.model.name
    in_ch = cfg.model.in_channels
    out_ch = cfg.model.out_channels

    if name == "monai_unet":
        return UNet(
            spatial_dims=3,
            in_channels=in_ch,
            out_channels=out_ch,
            channels=(16, 32, 64, 128, 256),
            strides=(2, 2, 2, 2),
            num_res_units=2,
        )

    if name == "swin_unetr":
        # use_checkpoint=True: gradient checkpointing -> tiết kiệm VRAM (cần cho GPU 8GB),
        # đổi lại train chậm hơn chút.
        kwargs = dict(
            in_channels=in_ch, out_channels=out_ch, feature_size=48, use_checkpoint=True
        )
        # MONAI >=1.5 bỏ tham số img_size; MONAI <=1.4 vẫn cần. Kiểm tra chữ ký
        # constructor để quyết định — tránh dùng try/except TypeError vì nó có thể
        # NUỐT một TypeError thật (vd tham số sai) rồi báo lỗi khác gây khó debug.
        if "img_size" in inspect.signature(SwinUNETR).parameters:
            kwargs["img_size"] = tuple(cfg.data.roi_size)
        return SwinUNETR(**kwargs)

    raise ValueError(
        f"Model không hỗ trợ: '{name}'. Chọn: monai_unet | swin_unetr"
    )


def count_parameters(model: nn.Module) -> int:
    """Đếm số tham số train được — dùng cho bảng benchmark."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

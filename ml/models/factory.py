"""Factory tạo model theo tên trong config — dùng chung cho train/eval/benchmark.

Hỗ trợ:
  - monai_unet : U-Net 3D của MONAI (baseline mạnh, ổn định).
  - swin_unetr : SwinUNETR — transformer encoder (điểm nhấn kiến trúc hiện đại).
  - unet_custom: U-Net tự viết ban đầu (baseline 2D, để đối chứng).
"""
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
        return SwinUNETR(
            img_size=tuple(cfg.data.roi_size),
            in_channels=in_ch,
            out_channels=out_ch,
            feature_size=48,
        )

    if name == "unet_custom":
        # Baseline 2D tự viết (giữ để đối chứng). Lưu ý: chỉ chạy với pipeline 2D-slice.
        from ml.training.model import UNet as CustomUNet

        return CustomUNet(in_channels=in_ch, out_channels=out_ch)

    raise ValueError(
        f"Model không hỗ trợ: '{name}'. Chọn: monai_unet | swin_unetr | unet_custom"
    )


def count_parameters(model: nn.Module) -> int:
    """Đếm số tham số train được — dùng cho bảng benchmark."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

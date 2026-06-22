"""MONAI transform pipelines cho dữ liệu CT 3D (NIfTI).

Train: resample spacing -> HU windowing -> crop foreground -> lấy patch ngẫu nhiên
cân bằng pos/neg -> augmentation nhẹ. Val/inference: bỏ phần ngẫu nhiên.
"""
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    Orientationd,
    Spacingd,
    ScaleIntensityRanged,
    CropForegroundd,
    RandCropByPosNegLabeld,
    RandFlipd,
    RandRotate90d,
    RandScaleIntensityd,
    RandShiftIntensityd,
    RandGaussianNoised,
    RandGaussianSmoothd,
    RandAdjustContrastd,
    EnsureTyped,
)

KEYS = ["image", "label"]


def _base_transforms(cfg):
    return [
        LoadImaged(keys=KEYS),
        EnsureChannelFirstd(keys=KEYS),
        Orientationd(keys=KEYS, axcodes="RAS"),
        Spacingd(keys=KEYS, pixdim=tuple(cfg.data.spacing), mode=("bilinear", "nearest")),
        ScaleIntensityRanged(
            keys="image",
            a_min=cfg.data.a_min,
            a_max=cfg.data.a_max,
            b_min=0.0,
            b_max=1.0,
            clip=True,
        ),
        CropForegroundd(keys=KEYS, source_key="image"),
    ]


def get_transforms(cfg, train: bool = True) -> Compose:
    tfs = _base_transforms(cfg)
    if train:
        tfs += [
            RandCropByPosNegLabeld(
                keys=KEYS,
                label_key="label",
                spatial_size=tuple(cfg.data.roi_size),
                pos=1,
                neg=1,
                num_samples=cfg.data.num_samples,
                image_key="image",
                image_threshold=0,
            ),
            # --- Augmentation nhẹ (mặc định, đã chứng minh hoạt động) ---
            RandFlipd(keys=KEYS, prob=0.2, spatial_axis=0),
            RandFlipd(keys=KEYS, prob=0.2, spatial_axis=1),
            RandFlipd(keys=KEYS, prob=0.2, spatial_axis=2),
            RandRotate90d(keys=KEYS, prob=0.2, max_k=3),
            RandShiftIntensityd(keys="image", offsets=0.1, prob=0.5),
        ]
        # --- Augmentation mạnh (TÙY CHỌN: data.strong_aug=true). Đã thử -> hại trên
        #     dataset nhỏ này (làm collapse sâu hơn). Giữ lại để thử nghiệm sau. ---
        if getattr(cfg.data, "strong_aug", False):
            tfs += [
                RandScaleIntensityd(keys="image", factors=0.1, prob=0.5),
                RandGaussianNoised(keys="image", prob=0.15, mean=0.0, std=0.01),
                RandGaussianSmoothd(keys="image", prob=0.15),
                RandAdjustContrastd(keys="image", prob=0.15, gamma=(0.7, 1.5)),
            ]
    tfs.append(EnsureTyped(keys=KEYS))
    return Compose(tfs)

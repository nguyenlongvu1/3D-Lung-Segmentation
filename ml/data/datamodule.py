"""Chia dữ liệu Medical Segmentation Decathlon thành train/val/test (MONAI).

Đọc danh sách ca CÓ NHÃN từ dataset.json rồi chia 3 phần CỐ ĐỊNH theo seed:
  - train: để học
  - val:   early stopping, chọn best model
  - test:  KHÓA LẠI, chỉ dùng để báo cáo cuối (model không bao giờ thấy khi train)

Cùng một cfg.seed => train3d và evaluate3d luôn tạo ra ĐÚNG CÙNG một split,
nên tập test thực sự "chưa từng thấy". Dữ liệu phải được tải trước bằng
`python -m ml.data.download`.
"""
import os
import random

from monai.data import (
    CacheDataset,
    DataLoader,
    list_data_collate,
    load_decathlon_datalist,
)

from .transforms import get_transforms


def _data_dir(cfg):
    return os.path.join(cfg.data.root, cfg.data.task)


def split_datalist(cfg):
    """Chia danh sách ca có nhãn thành (train, val, test) — cố định theo cfg.seed."""
    json_path = os.path.join(_data_dir(cfg), "dataset.json")
    if not os.path.exists(json_path):
        raise FileNotFoundError(
            f"Chưa thấy {json_path}. Tải dữ liệu trước: python -m ml.data.download"
        )
    items = list(
        load_decathlon_datalist(
            json_path, is_segmentation=True, data_list_key="training", base_dir=_data_dir(cfg)
        )
    )
    random.Random(cfg.seed).shuffle(items)  # xáo trộn tất định theo seed

    n = len(items)
    n_test = int(round(n * cfg.data.test_frac))
    n_val = int(round(n * cfg.data.val_frac))
    test = items[:n_test]
    val = items[n_test : n_test + n_val]
    train = items[n_test + n_val :]
    return train, val, test


def _cache_ds(cfg, datalist, train):
    return CacheDataset(
        data=datalist,
        transform=get_transforms(cfg, train=train),
        cache_rate=cfg.data.cache_rate,
        num_workers=cfg.data.num_workers,
    )


def get_loaders(cfg):
    """Trả về (train_loader, val_loader) cho huấn luyện."""
    train_list, val_list, test_list = split_datalist(cfg)
    print(f"Split (seed={cfg.seed}): train={len(train_list)} | val={len(val_list)} | test={len(test_list)}")

    loader_workers = int(getattr(cfg.data, "loader_workers", 0))
    train_loader = DataLoader(
        _cache_ds(cfg, train_list, train=True),
        batch_size=cfg.data.batch_size,
        shuffle=True,
        num_workers=loader_workers,
        collate_fn=list_data_collate,
        pin_memory=True,
    )
    val_loader = DataLoader(
        _cache_ds(cfg, val_list, train=False),
        batch_size=1,
        shuffle=False,
        num_workers=loader_workers,
        pin_memory=True,
    )
    return train_loader, val_loader


def get_test_loader(cfg):
    """Trả về DataLoader cho tập test (đánh giá cuối)."""
    _, _, test_list = split_datalist(cfg)
    return DataLoader(
        _cache_ds(cfg, test_list, train=False),
        batch_size=1,
        shuffle=False,
        num_workers=int(getattr(cfg.data, "loader_workers", 0)),
        pin_memory=True,
    )

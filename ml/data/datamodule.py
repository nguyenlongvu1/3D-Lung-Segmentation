"""Tạo DataLoader train/val từ Medical Segmentation Decathlon (MONAI).

DecathlonDataset tự tải dữ liệu (nếu download=True), tách train/val theo val_frac,
và cache mẫu để tăng tốc. Val dùng batch_size=1 vì inference theo sliding-window
trên toàn volume.
"""
from monai.apps import DecathlonDataset
from monai.data import DataLoader, list_data_collate

from .transforms import get_transforms


def get_datasets(cfg, download: bool = True):
    train_tf = get_transforms(cfg, train=True)
    val_tf = get_transforms(cfg, train=False)

    train_ds = DecathlonDataset(
        root_dir=cfg.data.root,
        task=cfg.data.task,
        section="training",
        transform=train_tf,
        download=download,
        val_frac=cfg.data.val_frac,
        cache_rate=cfg.data.cache_rate,
        num_workers=cfg.data.num_workers,
    )
    val_ds = DecathlonDataset(
        root_dir=cfg.data.root,
        task=cfg.data.task,
        section="validation",
        transform=val_tf,
        download=False,  # đã tải ở lần tạo train_ds
        val_frac=cfg.data.val_frac,
        cache_rate=cfg.data.cache_rate,
        num_workers=cfg.data.num_workers,
    )
    return train_ds, val_ds


def get_loaders(cfg, download: bool = True):
    train_ds, val_ds = get_datasets(cfg, download=download)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.data.batch_size,
        shuffle=True,
        num_workers=cfg.data.num_workers,
        collate_fn=list_data_collate,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=cfg.data.num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader

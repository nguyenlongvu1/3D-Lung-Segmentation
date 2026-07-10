"""Test pipeline transforms 3D: cấu trúc train/val, toggle strong_aug, chạy thực tế."""
from monai.transforms import Compose

from ml.config import load_config
from ml.data.transforms import get_transforms


def _small_cfg():
    cfg = load_config()
    cfg.data.roi_size = [16, 16, 16]
    cfg.data.num_samples = 2
    return cfg


def test_val_transforms_no_random_crop():
    """Val (train=False) KHÔNG có bước lấy patch ngẫu nhiên/augment."""
    tfs = get_transforms(load_config(), train=False)
    assert isinstance(tfs, Compose)
    names = [t.__class__.__name__ for t in tfs.transforms]
    assert "RandCropByPosNegLabeld" not in names
    assert not any(n.startswith("Rand") for n in names)
    assert names[-1] == "EnsureTyped"


def test_train_transforms_add_augmentation():
    """Train (train=False -> True) THÊM crop patch + các phép flip/rotate/shift."""
    n_val = len(get_transforms(load_config(), train=False).transforms)
    n_train = len(get_transforms(load_config(), train=True).transforms)
    assert n_train > n_val
    names = [t.__class__.__name__ for t in get_transforms(load_config(), train=True).transforms]
    assert "RandCropByPosNegLabeld" in names
    assert "RandFlipd" in names


def test_strong_aug_toggle_adds_transforms():
    """strong_aug=True thêm đúng 4 phép augmentation mạnh so với mặc định."""
    cfg = load_config()
    cfg.data.strong_aug = False
    base = len(get_transforms(cfg, train=True).transforms)
    cfg.data.strong_aug = True
    strong = len(get_transforms(cfg, train=True).transforms)
    assert strong == base + 4


def test_val_pipeline_runs_and_normalizes(make_nifti):
    """Chạy val transforms thật: ảnh scale về [0,1], giữ nguyên cặp image/label."""
    img_path, lbl_path = make_nifti()
    out = get_transforms(_small_cfg(), train=False)({"image": img_path, "label": lbl_path})
    assert out["image"].shape == out["label"].shape
    assert out["image"].shape[0] == 1  # channel-first
    assert 0.0 <= float(out["image"].min())
    assert float(out["image"].max()) <= 1.0


def test_train_pipeline_yields_patches(make_nifti):
    """Train transforms trả về num_samples patch đúng kích thước roi."""
    cfg = _small_cfg()
    img_path, lbl_path = make_nifti()
    patches = get_transforms(cfg, train=True)({"image": img_path, "label": lbl_path})
    assert isinstance(patches, list)
    assert len(patches) == cfg.data.num_samples
    assert tuple(patches[0]["image"].shape) == (1, 16, 16, 16)

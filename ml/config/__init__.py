"""Load YAML config thành object truy cập bằng dấu chấm (cfg.data.roi_size)."""
import os
from types import SimpleNamespace

import yaml

_DEFAULT = os.path.join(os.path.dirname(__file__), "default.yaml")


def _to_ns(obj):
    if isinstance(obj, dict):
        return SimpleNamespace(**{k: _to_ns(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [_to_ns(v) for v in obj]
    return obj


def _to_dict(obj):
    """Chuyển ngược SimpleNamespace -> dict thuần (để lưu lại config của mỗi run)."""
    if isinstance(obj, SimpleNamespace):
        return {k: _to_dict(v) for k, v in vars(obj).items()}
    if isinstance(obj, list):
        return [_to_dict(v) for v in obj]
    return obj


def load_config(path: str | None = None) -> SimpleNamespace:
    """Đọc file YAML cấu hình; mặc định dùng ml/config/default.yaml."""
    with open(path or _DEFAULT, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return _to_ns(raw)


def save_config(cfg: SimpleNamespace, path: str) -> None:
    """Lưu lại cấu hình (snapshot) cho khả năng tái lập của một run."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(_to_dict(cfg), f, sort_keys=False, allow_unicode=True)

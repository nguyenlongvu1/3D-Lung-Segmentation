"""Test model factory: build đúng kiến trúc, đếm params, báo lỗi tên sai."""
import pytest
import torch.nn as nn

from ml.config import load_config
from ml.models.factory import build_model, count_parameters


def test_build_monai_unet():
    cfg = load_config()
    cfg.model.name = "monai_unet"
    model = build_model(cfg)
    assert isinstance(model, nn.Module)
    assert count_parameters(model) > 0


def test_unknown_model_raises():
    cfg = load_config()
    cfg.model.name = "khong_ton_tai"
    with pytest.raises(ValueError):
        build_model(cfg)

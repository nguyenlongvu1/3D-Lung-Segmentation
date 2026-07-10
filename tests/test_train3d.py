"""Test build_loss của train3d: đúng loại loss theo tên, mặc định, và báo lỗi tên sai."""
import pytest
from monai.losses import DiceCELoss, DiceFocalLoss, TverskyLoss

from ml.training.train3d import build_loss


def test_build_each_loss_type():
    assert isinstance(build_loss("tversky"), TverskyLoss)
    assert isinstance(build_loss("dicefocal"), DiceFocalLoss)
    assert isinstance(build_loss("dicece"), DiceCELoss)


def test_build_loss_case_insensitive():
    assert isinstance(build_loss("DiceFocal"), DiceFocalLoss)


def test_build_loss_default_is_tversky():
    assert isinstance(build_loss(None), TverskyLoss)


def test_build_loss_unknown_raises():
    with pytest.raises(ValueError):
        build_loss("khong_ton_tai")

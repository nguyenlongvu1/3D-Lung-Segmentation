"""Test LungTumorInference end-to-end trên CPU với model + input tổng hợp.

Không cần checkpoint đã train: khởi tạo UNet, lưu state_dict vào file tạm, rồi
chạy predict trên 1 ảnh NIfTI giả. Kiểm tra hợp đồng đầu ra (shape/dtype/thể tích).
"""
import numpy as np
import torch

from ml.config import load_config
from ml.models.factory import build_model
from api.inference import LungTumorInference, get_infer_transforms


def _small_cfg():
    cfg = load_config()
    cfg.model.name = "monai_unet"
    cfg.data.roi_size = [32, 32, 32]  # >=32 để instance-norm không nghẽn ở đáy UNet
    return cfg


def _make_engine(tmp_path, cfg):
    ckpt = tmp_path / "best.pth"
    torch.save(build_model(cfg).state_dict(), str(ckpt))
    return LungTumorInference(str(ckpt), cfg=cfg, device=torch.device("cpu"))


def test_infer_transforms_normalize(make_nifti):
    img_path = make_nifti(with_label=True)[0]
    out = get_infer_transforms(_small_cfg())(img_path)
    assert out.shape[0] == 1  # channel-first
    assert 0.0 <= float(out.min()) and float(out.max()) <= 1.0


def test_predict_output_contract(tmp_path, make_nifti):
    cfg = _small_cfg()
    engine = _make_engine(tmp_path, cfg)
    img_path = make_nifti(shape=(72, 72, 72), with_label=False)
    result = engine.predict(img_path)
    assert set(result) == {"image", "mask", "tumor_ml"}
    assert result["image"].shape == result["mask"].shape
    assert result["mask"].dtype == np.uint8
    assert set(np.unique(result["mask"])).issubset({0, 1})
    assert isinstance(result["tumor_ml"], float)
    assert result["tumor_ml"] >= 0.0


def test_voxel_volume_from_spacing(tmp_path):
    cfg = _small_cfg()
    engine = _make_engine(tmp_path, cfg)
    # voxel_mm3 = tích 3 chiều spacing
    assert engine.voxel_mm3 == float(np.prod(cfg.data.spacing))

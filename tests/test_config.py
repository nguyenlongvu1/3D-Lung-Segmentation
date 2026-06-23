"""Test load/save config (dot-access + round-trip)."""
from ml.config import load_config, save_config


def test_defaults_loaded():
    cfg = load_config()
    # vài giá trị neo quan trọng của pipeline phổi
    assert cfg.data.a_min == -1000
    assert cfg.data.a_max == 400
    assert len(cfg.data.spacing) == 3
    assert len(cfg.data.roi_size) == 3
    assert cfg.model.out_channels == 2          # nền + u
    assert 0.0 < cfg.data.val_frac < 1.0


def test_save_load_roundtrip(tmp_path):
    cfg = load_config()
    out = tmp_path / "cfg.yaml"
    save_config(cfg, str(out))
    cfg2 = load_config(str(out))
    assert cfg2.model.name == cfg.model.name
    assert cfg2.data.roi_size == cfg.data.roi_size
    assert cfg2.train.loss == cfg.train.loss

"""Test evaluate3d: ghi CSV (header 1 lần, append), và báo lỗi khi thiếu checkpoint."""
import csv

import pytest

from ml.config import load_config
from ml.training.evaluate3d import _append_csv, evaluate


def test_append_csv_writes_header_once(tmp_path):
    path = tmp_path / "sub" / "benchmark.csv"  # thư mục con chưa tồn tại -> phải tự tạo
    row1 = {"model": "monai_unet", "dice": 0.36}
    row2 = {"model": "swin_unetr", "dice": 0.42}
    _append_csv(str(path), row1)
    _append_csv(str(path), row2)
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 2  # header chỉ 1 dòng, không lặp
    assert rows[0]["model"] == "monai_unet"
    assert rows[1]["model"] == "swin_unetr"


def test_evaluate_missing_checkpoint_raises(tmp_path):
    cfg = load_config()
    cfg.train.ckpt_dir = str(tmp_path / "no_ckpt")
    with pytest.raises(FileNotFoundError):
        evaluate(cfg)

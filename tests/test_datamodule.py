"""Test chia dữ liệu: split cố định theo seed, đúng tỉ lệ, không rò rỉ giữa các tập."""
import json
import os

import pytest

from ml.config import load_config
from ml.data.datamodule import _data_dir, split_datalist


def _make_dataset(tmp_path, n=10):
    """Tạo cây thư mục MSD tối thiểu (dataset.json + file rỗng) cho n ca có nhãn."""
    cfg = load_config()
    cfg.data.root = str(tmp_path)
    cfg.data.task = "Task_Fake"
    cfg.seed = 42
    cfg.data.val_frac = 0.2
    cfg.data.test_frac = 0.2
    task_dir = tmp_path / "Task_Fake"
    (task_dir / "imagesTr").mkdir(parents=True)
    (task_dir / "labelsTr").mkdir(parents=True)
    training = []
    for i in range(n):
        rel_img, rel_lbl = f"imagesTr/c{i}.nii.gz", f"labelsTr/c{i}.nii.gz"
        (task_dir / rel_img).write_text("")
        (task_dir / rel_lbl).write_text("")
        training.append({"image": rel_img, "label": rel_lbl})
    (task_dir / "dataset.json").write_text(json.dumps({"training": training}))
    return cfg


def test_data_dir_joins_root_and_task():
    cfg = load_config()
    cfg.data.root = "/data"
    cfg.data.task = "Task06_Lung"
    assert _data_dir(cfg) == os.path.join("/data", "Task06_Lung")


def test_split_sizes_match_fractions(tmp_path):
    cfg = _make_dataset(tmp_path, n=10)
    train, val, test = split_datalist(cfg)
    assert (len(train), len(val), len(test)) == (6, 2, 2)


def test_split_is_deterministic_and_disjoint(tmp_path):
    cfg = _make_dataset(tmp_path, n=10)
    train, val, test = split_datalist(cfg)
    train2, val2, test2 = split_datalist(cfg)
    # cùng seed -> đúng cùng một split
    assert [x["image"] for x in test] == [x["image"] for x in test2]
    assert [x["image"] for x in val] == [x["image"] for x in val2]
    # 3 tập rời nhau, phủ hết
    all_imgs = [x["image"] for x in train + val + test]
    assert len(set(all_imgs)) == 10


def test_different_seed_changes_split(tmp_path):
    cfg = _make_dataset(tmp_path, n=10)
    _, _, test = split_datalist(cfg)
    cfg.seed = 123
    _, _, test_b = split_datalist(cfg)
    assert [x["image"] for x in test] != [x["image"] for x in test_b]


def test_missing_dataset_json_raises(tmp_path):
    cfg = load_config()
    cfg.data.root = str(tmp_path)
    cfg.data.task = "khong_ton_tai"
    with pytest.raises(FileNotFoundError):
        split_datalist(cfg)

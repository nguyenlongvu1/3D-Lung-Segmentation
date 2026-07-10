"""Fixtures dùng chung cho test: tạo file NIfTI tổng hợp (không cần tải data thật).

Nhờ vậy các test cho transforms / datamodule / inference chạy được GPU-free và
KHÔNG phụ thuộc dataset MSD (~9GB) hay checkpoint đã train.
"""
import numpy as np
import nibabel as nib
import pytest


@pytest.fixture
def make_nifti(tmp_path):
    """Trả về hàm tạo cặp (image, label) .nii.gz tổng hợp trong tmp_path.

    image: giá trị HU ngẫu nhiên trong [a_min, a_max]; label: 1 khối lập phương nhỏ.
    """
    counter = {"n": 0}

    def _make(shape=(72, 72, 72), label_box=(20, 40), with_label=True):
        i = counter["n"]
        counter["n"] += 1
        rng = np.random.default_rng(i)
        img = rng.integers(-1000, 400, size=shape).astype(np.int16)
        img_path = tmp_path / f"img_{i}.nii.gz"
        nib.save(nib.Nifti1Image(img, np.eye(4)), str(img_path))
        if not with_label:
            return str(img_path)
        lbl = np.zeros(shape, dtype=np.uint8)
        lo, hi = label_box
        lbl[lo:hi, lo:hi, lo:hi] = 1
        lbl_path = tmp_path / f"lbl_{i}.nii.gz"
        nib.save(nib.Nifti1Image(lbl, np.eye(4)), str(lbl_path))
        return str(img_path), str(lbl_path)

    return _make

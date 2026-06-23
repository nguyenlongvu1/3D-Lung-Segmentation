"""Test helper render của API (không cần model/checkpoint)."""
import numpy as np

from api.main import _best_slice, _render_overlay


def test_best_slice_picks_max_tumor():
    mask = np.zeros((4, 4, 6), dtype=np.uint8)
    mask[:, :, 4] = 1                         # u tập trung ở lát z=4
    assert _best_slice(mask, mask.shape[2]) == 4


def test_best_slice_empty_returns_middle():
    mask = np.zeros((4, 4, 6), dtype=np.uint8)
    assert _best_slice(mask, 6) == 3          # không có u -> lát giữa = 6 // 2


def test_render_overlay_returns_png():
    img = np.random.rand(8, 8).astype(np.float32)
    mask = np.zeros((8, 8), dtype=np.uint8)
    mask[2:5, 2:5] = 1
    buf = _render_overlay(img, mask)
    data = buf.getvalue()
    assert data[:8] == b"\x89PNG\r\n\x1a\n"   # magic bytes của PNG

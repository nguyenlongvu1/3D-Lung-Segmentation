"""Test onnx_export: _bench đếm/định thời đúng; export+verify (bỏ qua nếu thiếu onnx)."""
import pytest
import torch

from ml.config import load_config
from ml.models.factory import build_model
from ml.deploy.onnx_export import _bench, export_to_onnx, verify


def test_bench_runs_iters_plus_warmup_and_returns_ms():
    calls = {"n": 0}

    def fn():
        calls["n"] += 1

    ms = _bench(fn, iters=5, warmup=2)
    assert calls["n"] == 7          # warmup + iters
    assert isinstance(ms, float)
    assert ms >= 0.0


def test_export_and_verify_roundtrip(tmp_path):
    """Export ONNX rồi so nhãn argmax PyTorch vs ONNX Runtime (cần onnx/onnxruntime)."""
    pytest.importorskip("onnx")
    ort = pytest.importorskip("onnxruntime")

    cfg = load_config()
    cfg.model.name = "monai_unet"
    roi = (32, 32, 32)
    device = torch.device("cpu")
    model = build_model(cfg)
    model.eval()

    path = str(tmp_path / "model.onnx")
    export_to_onnx(model, roi, path, device)

    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    assert verify(model, sess, roi, device) is True

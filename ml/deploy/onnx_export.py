"""Export UNet 3D -> ONNX, kiểm tra số học, và benchmark latency (PyTorch vs ONNX Runtime).

Tại sao:
  - ONNX tách model khỏi PyTorch -> chạy được bằng ONNX Runtime trên máy CPU-only / edge,
    KHÔNG cần cài CUDA/PyTorch -> deploy nhẹ & rẻ hơn.
  - Benchmark cho con số deployability cụ thể để lập luận (xem mục accuracy-vs-deployability).

Đo ở mức 1 patch roi (vd 96^3) — đúng đơn vị mà sliding-window inference gọi model.

    python -m ml.deploy.onnx_export
    python -m ml.deploy.onnx_export --iters 50
"""
import argparse
import csv
import os
import time

import numpy as np
import torch

from ml.config import load_config
from ml.models.factory import build_model


def export_to_onnx(model, roi, path, device):
    """Trace model với input cố định [1,1,*roi] (batch để động) -> file .onnx."""
    model.eval()
    dummy = torch.randn(1, 1, *roi, device=device)
    with torch.no_grad():
        torch.onnx.export(
            model,
            dummy,
            path,
            input_names=["input"],
            output_names=["logits"],
            dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
            opset_version=17,
        )
    print(f"✅ Đã export ONNX -> {path}")


def verify(model, sess, roi, device):
    """So PyTorch vs ONNX Runtime. Với segmentation, phép kiểm tra ĐÚNG là nhãn sau
    argmax có trùng không (chênh lệch logits nhỏ không đổi kết quả phân đoạn)."""
    x = torch.randn(1, 1, *roi)
    with torch.no_grad():
        torch_out = model(x.to(device)).cpu().numpy()
    ort_out = sess.run(None, {"input": x.numpy()})[0]
    max_diff = float(np.abs(torch_out - ort_out).max())
    agree = float((torch_out.argmax(1) == ort_out.argmax(1)).mean()) * 100.0
    ok = agree > 99.9
    print(
        f"Kiểm tra: max|Δ logits| = {max_diff:.2e} | nhãn argmax trùng {agree:.3f}% "
        f"-> {'OK' if ok else 'KHÁC BIỆT'}"
    )
    return ok


def _bench(fn, iters, warmup, cuda=False):
    """Trả về ms trung bình mỗi lần chạy (có warmup + đồng bộ CUDA nếu cần)."""
    for _ in range(warmup):
        fn()
    if cuda:
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    if cuda:
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1000.0


def main():
    p = argparse.ArgumentParser(description="Export ONNX + benchmark latency")
    p.add_argument("--config", default=None)
    p.add_argument("--iters", type=int, default=20, help="số lần đo mỗi backend")
    p.add_argument("--warmup", type=int, default=5)
    args = p.parse_args()

    cfg = load_config(args.config)
    roi = tuple(cfg.data.roi_size)
    ckpt = os.path.join(cfg.train.ckpt_dir, f"best_{cfg.model.name}.pth")
    if not os.path.exists(ckpt):
        raise FileNotFoundError(f"Chưa có checkpoint: {ckpt}. Hãy train trước (train3d).")
    onnx_path = os.path.join(cfg.train.ckpt_dir, f"{cfg.model.name}.onnx")
    cpu = torch.device("cpu")

    # --- Export & verify CÙNG trên CPU (để so sánh số học công bằng, loại nhiễu GPU↔CPU) ---
    model_cpu = build_model(cfg)
    model_cpu.load_state_dict(torch.load(ckpt, map_location="cpu"))
    model_cpu.eval()
    export_to_onnx(model_cpu, roi, onnx_path, cpu)

    import onnxruntime as ort

    sess_cpu = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    verify(model_cpu, sess_cpu, roi, cpu)

    # --- Benchmark: cùng 1 patch roi qua nhiều backend ---
    x_np = np.random.randn(1, 1, *roi).astype(np.float32)
    x_cpu = torch.from_numpy(x_np)

    results = {}
    with torch.no_grad():
        if torch.cuda.is_available():
            model_gpu = build_model(cfg)
            model_gpu.load_state_dict(torch.load(ckpt, map_location="cpu"))
            model_gpu.to("cuda").eval()
            x_gpu = x_cpu.to("cuda")
            results["PyTorch (CUDA)"] = _bench(
                lambda: model_gpu(x_gpu), args.iters, args.warmup, cuda=True
            )
        results["PyTorch (CPU)"] = _bench(lambda: model_cpu(x_cpu), args.iters, args.warmup)
    results["ONNX Runtime (CPU)"] = _bench(
        lambda: sess_cpu.run(None, {"input": x_np}), args.iters, args.warmup
    )

    # --- In bảng + ghi CSV ---
    print(f"\n=== Latency / patch {roi} (trung bình {args.iters} lần) ===")
    base = results.get("PyTorch (CPU)")
    for name, ms in results.items():
        extra = ""
        if base and name == "ONNX Runtime (CPU)":
            extra = f"  ({base / ms:.2f}x so với PyTorch CPU)"
        print(f"  {name:22s}: {ms:8.2f} ms{extra}")

    out_csv = os.path.join("results", "latency_benchmark.csv")
    os.makedirs("results", exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["backend", "ms_per_patch", "roi", "iters"])
        for name, ms in results.items():
            w.writerow([name, round(ms, 2), "x".join(map(str, roi)), args.iters])
    print(f"\nĐã ghi -> {out_csv}")


if __name__ == "__main__":
    main()

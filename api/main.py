"""FastAPI demo: upload CT (.nii.gz) -> segment khối u phổi 3D -> viewer lát cắt.

Luồng:
  POST /predict   nhận file NIfTI, chạy inference, lưu kết quả trong RAM, trả metadata.
  GET  /slice/{z} trả PNG lát cắt axial z (ảnh CT + overlay khối u màu đỏ).

Lưu ý: trạng thái lưu trong biến toàn cục (demo 1 người dùng), không dùng cho production.
"""
import io
import logging
import os
import tempfile

import numpy as np
from fastapi import FastAPI, File, UploadFile, Request, HTTPException
from fastapi.responses import StreamingResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from PIL import Image

logger = logging.getLogger("lung_seg.api")

app = FastAPI(title="Lung Tumor Segmentation 3D")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CKPT_PATH = os.environ.get(
    "CKPT_PATH", os.path.join(BASE_DIR, "checkpoints", "best_monai_unet.pth")
)

app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "web", "static")), name="static")
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "web", "templates"))

_engine = None      # nạp model lười (lần predict đầu tiên)
_last = {}          # kết quả gần nhất: {"image": [H,W,D], "mask": [H,W,D]}


def get_engine():
    global _engine
    if _engine is None:
        if not os.path.exists(CKPT_PATH):
            raise HTTPException(503, f"Chưa có checkpoint: {CKPT_PATH}. Hãy train trước.")
        from api.inference import LungTumorInference  # import lười để app khởi động nhanh

        _engine = LungTumorInference(CKPT_PATH)
    return _engine


def _best_slice(mask, n):
    """Lát có nhiều khối u nhất (để hiển thị mặc định), hoặc lát giữa."""
    return int(mask.sum(axis=(0, 1)).argmax()) if mask.sum() > 0 else n // 2


def _render_overlay(img2d, mask2d):
    """Ghép ảnh CT xám + overlay khối u đỏ -> PNG."""
    gray = (np.clip(img2d, 0, 1) * 255).astype(np.uint8)
    rgb = np.stack([gray, gray, gray], axis=-1)
    fg = mask2d.astype(bool)
    rgb[fg] = (0.5 * rgb[fg] + 0.5 * np.array([255, 0, 0])).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="PNG")
    buf.seek(0)
    return buf


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.filename or not file.filename.endswith((".nii", ".nii.gz")):
        raise HTTPException(400, "Chỉ nhận file .nii hoặc .nii.gz")

    data = await file.read()
    suffix = ".nii.gz" if file.filename.endswith(".gz") else ".nii"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(data)
        tmp_path = tmp.name
    try:
        result = get_engine().predict(tmp_path)
    except HTTPException:
        raise
    except (ValueError, RuntimeError, OSError) as e:
        # Lỗi do file đầu vào hỏng/không đọc được (NIfTI lỗi, shape sai...) -> lỗi client.
        # Vẫn log đầy đủ để debug thay vì chỉ nuốt traceback.
        logger.warning("Không đọc/xử lý được file '%s': %s", file.filename, e, exc_info=True)
        raise HTTPException(400, f"Không xử lý được file: {e}")
    except Exception as e:  # noqa: BLE001
        # Lỗi bất ngờ phía server (bug, hết bộ nhớ...) -> KHÔNG che thành 400.
        # Ghi log traceback đầy đủ và trả 500 để lỗi thật được thấy, không bị nuốt.
        logger.exception("Lỗi không mong đợi khi inference file '%s'", file.filename)
        raise HTTPException(500, f"Lỗi máy chủ khi xử lý file: {e}")
    finally:
        try:
            os.unlink(tmp_path)
        except OSError as e:
            # Dọn file tạm thất bại không nên che lấp kết quả/lỗi của request.
            logger.warning("Không xóa được file tạm '%s': %s", tmp_path, e)

    _last["image"], _last["mask"] = result["image"], result["mask"]
    n = int(result["image"].shape[2])
    return JSONResponse(
        {
            "num_slices": n,
            "default_slice": _best_slice(result["mask"], n),
            "tumor_ml": round(result["tumor_ml"], 2),
        }
    )


@app.get("/slice/{z}")
def get_slice(z: int):
    if "image" not in _last:
        raise HTTPException(404, "Chưa có kết quả. Hãy upload file trước.")
    img, mask = _last["image"], _last["mask"]
    z = max(0, min(z, img.shape[2] - 1))
    return StreamingResponse(_render_overlay(img[:, :, z], mask[:, :, z]), media_type="image/png")

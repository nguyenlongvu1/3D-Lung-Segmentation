from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import StreamingResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import torch
import os
from .model import load_model
from .utils import preprocess_image, postprocess_mask

# --- Khởi tạo ---
app = FastAPI(title="Lung Segmentation App")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "unet_lung_seg.pth")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

model = load_model(MODEL_PATH, DEVICE)

app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "web", "static")), name="static")
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "web", "templates"))


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    tensor = preprocess_image(file.file)
    with torch.no_grad():
        pred = model(tensor.to(DEVICE))
    mask_buf = postprocess_mask(pred)
    return StreamingResponse(mask_buf, media_type="image/png")

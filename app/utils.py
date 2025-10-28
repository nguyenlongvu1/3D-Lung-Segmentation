import torch
import numpy as np
from PIL import Image
import io

def preprocess_image(uploaded_file, img_size=256):
    """Chuyển file ảnh upload thành tensor phù hợp cho model"""
    image = Image.open(uploaded_file).convert("L")
    image = image.resize((img_size, img_size))
    img_array = np.array(image, dtype=np.float32) / 255.0
    tensor = torch.tensor(img_array).unsqueeze(0).unsqueeze(0)
    return tensor

def postprocess_mask(pred_tensor):
    """Chuyển output tensor -> ảnh mask (grayscale PNG)"""
    mask = (pred_tensor.squeeze().cpu().numpy() > 0.5).astype(np.uint8) * 255
    mask_img = Image.fromarray(mask)
    buf = io.BytesIO()
    mask_img.save(buf, format="PNG")
    buf.seek(0)
    return buf

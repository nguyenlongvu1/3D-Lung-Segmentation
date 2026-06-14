import torch
from PIL import Image
import io

def preprocess_image(uploaded_file, img_size=256):
    """Chuyển file ảnh upload thành tensor phù hợp cho model"""
    image = Image.open(uploaded_file).convert("L")
    image = image.resize((img_size, img_size))
    pixels = torch.tensor(list(image.getdata()), dtype=torch.float32)
    tensor = pixels.view(img_size, img_size).div(255.0).unsqueeze(0).unsqueeze(0)
    return tensor

def postprocess_mask(pred_tensor):
    """Chuyển output tensor -> ảnh mask (grayscale PNG)"""
    mask_tensor = pred_tensor.squeeze().detach().cpu().gt(0.5).to(torch.uint8).mul(255)
    h, w = mask_tensor.shape[-2], mask_tensor.shape[-1]
    mask_img = Image.new("L", (w, h))
    mask_img.putdata(mask_tensor.flatten().tolist())
    buf = io.BytesIO()
    mask_img.save(buf, format="PNG")
    buf.seek(0)
    return buf

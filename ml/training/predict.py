# predict.py
import torch
from PIL import Image
from torchvision import transforms
from .model import UNet
from .utils import visualize_results
from .config import Config

def predict_single(image_path, mask_path=None):
    config = Config()
    model = UNet(in_channels=1, out_channels=1).to(config.device)
    model.load_state_dict(torch.load(config.save_model_path, map_location=config.device))
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((config.img_size, config.img_size)),
        transforms.ToTensor(),
    ])

    img = Image.open(image_path).convert("L")
    img_tensor = transform(img).unsqueeze(0).to(config.device)

    with torch.no_grad():
        pred = model(img_tensor)
        pred_bin = (pred > 0.5).float()

    if mask_path:
        mask = Image.open(mask_path).convert("L")
        mask_tensor = transform(mask)
        visualize_results(img_tensor[0], mask_tensor, pred_bin[0])
    else:
        visualize_results(img_tensor[0], pred_bin[0], pred_bin[0])

if __name__ == "__main__":
    image_path = r"C:\Users\ADMIN\Unet\data\test\image\1011.png"
    mask_path = r"C:\Users\ADMIN\Unet\data\test\mask\1011.png"
    predict_single(image_path, mask_path)

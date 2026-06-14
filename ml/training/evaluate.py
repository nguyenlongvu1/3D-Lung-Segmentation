# evaluate.py
import torch
from torch.utils.data import DataLoader
from .datasets import LungSegDataset
from .model import UNet
from .utils import dice_coef, iou_score, visualize_results
from .config import Config
import os

def evaluate():
    config = Config()
    model = UNet(in_channels=1, out_channels=1).to(config.device)
    model.load_state_dict(torch.load(config.save_model_path, map_location=config.device))
    model.eval()

    test_data = LungSegDataset(config.test_img_dir, config.test_mask_dir, config.img_size, augment=False)
    test_loader = DataLoader(test_data, batch_size=1, shuffle=False)

    dice_total, iou_total = 0, 0
    with torch.no_grad():
        for idx, (img, mask) in enumerate(test_loader):
            img, mask = img.to(config.device), mask.to(config.device)
            pred = model(img)
            pred_bin = (pred > 0.5).float()

            dice_total += dice_coef(pred_bin, mask).item()
            iou_total += iou_score(pred_bin, mask).item()

            if idx < 3:
                visualize_results(img[0], mask[0], pred_bin[0])

    avg_dice = dice_total / len(test_loader)
    avg_iou = iou_total / len(test_loader)

    print(f"\n Final Test Dice: {avg_dice:.4f}, IoU: {avg_iou:.4f}")

    # Lưu kết quả
    os.makedirs("results", exist_ok=True)
    with open("results/test_metrics.txt", "w") as f:
        f.write(f"Test Dice: {avg_dice:.4f}\nTest IoU: {avg_iou:.4f}\n")
    print("Saved test results to results/test_metrics.txt")

if __name__ == "__main__":
    evaluate()

# utils.py
import torch
import matplotlib.pyplot as plt
import os

def dice_coef(pred, target, smooth=1e-6):
    pred = pred.view(-1)
    target = target.view(-1)
    intersection = (pred * target).sum()
    return (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)

def iou_score(pred, target, smooth=1e-6):
    pred = pred.view(-1)
    target = target.view(-1)
    intersection = (pred * target).sum()
    total = (pred + target).sum()
    union = total - intersection
    return (intersection + smooth) / (union + smooth)

def save_checkpoint(model, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")

def visualize_results(image, mask_true, mask_pred):
    image = image.squeeze().cpu().numpy()
    mask_true = mask_true.squeeze().cpu().numpy()
    mask_pred = mask_pred.squeeze().cpu().numpy()

    fig, ax = plt.subplots(1, 4, figsize=(12, 4))
    ax[0].imshow(image, cmap='gray'); ax[0].set_title("Input")
    ax[1].imshow(mask_true, cmap='gray'); ax[1].set_title("Ground Truth")
    ax[2].imshow(mask_pred, cmap='gray'); ax[2].set_title("Prediction")
    ax[3].imshow(image, cmap='gray'); ax[3].imshow(mask_pred, alpha=0.5, cmap='Reds'); ax[3].set_title("Overlay")
    plt.tight_layout()
    plt.show()


import torch
import torch.nn as nn
from tqdm import tqdm
from .utils import dice_coef, iou_score, save_checkpoint
from .visualize import plot_metrics

def train(model, train_loader, val_loader, optimizer, criterion, config):
    best_dice = 0.0
    train_losses, val_dices, val_ious = [], [], []

    patience = 6   
    counter = 0

    for epoch in range(config.num_epochs):
        model.train()
        train_loss = 0

        for images, masks in tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.num_epochs}"):
            images, masks = images.to(config.device), masks.to(config.device)
            preds = model(images)
            loss = criterion(preds, masks)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # ---- Validation ----
        model.eval()
        val_dice, val_iou = 0, 0
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(config.device), masks.to(config.device)
                preds = model(images)
                preds = (preds > 0.5).float()
                val_dice += dice_coef(preds, masks).item()
                val_iou += iou_score(preds, masks).item()

        val_dice /= len(val_loader)
        val_iou /= len(val_loader)
        train_loss /= len(train_loader)

        train_losses.append(train_loss)
        val_dices.append(val_dice)
        val_ious.append(val_iou)

        print(f"Epoch [{epoch+1}/{config.num_epochs}] | Loss: {train_loss:.4f} | Dice: {val_dice:.4f} | IoU: {val_iou:.4f}")

        # ---- Model checkpoint & early stopping ----
        if val_dice > best_dice:
            best_dice = val_dice
            save_checkpoint(model, config.save_model_path)
            counter = 0  # reset counter 
        else:
            counter += 1
            if counter >= patience:
                print(f"Early stopping at epoch {epoch+1} (no improvement for {patience} epochs)")
                break

    # ---- Plot learning curves ----
    plot_metrics(train_losses, val_dices, val_ious)

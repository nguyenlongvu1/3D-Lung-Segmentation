# training/config.py
import torch
import os

class Config:
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    # Paths
    train_img_dir = os.path.join(BASE_DIR, "data", "train", "image")
    train_mask_dir = os.path.join(BASE_DIR, "data", "train", "mask")
    val_img_dir = os.path.join(BASE_DIR, "data", "val", "image")
    val_mask_dir = os.path.join(BASE_DIR, "data", "val", "mask")
    test_img_dir = os.path.join(BASE_DIR, "data", "test", "image")
    test_mask_dir = os.path.join(BASE_DIR, "data", "test", "mask")
    save_model_path = os.path.join(BASE_DIR, "checkpoints", "best_model.pth")

    # Training
    img_size = 256
    batch_size = 4
    num_epochs = 30
    learning_rate = 1e-3

    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"



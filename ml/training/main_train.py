# main_train.py
import torch
from torch.utils.data import DataLoader
from .datasets import LungSegDataset
from .model import UNet
from .train import train
from .config import Config

def main():
    config = Config()
    print(f"Using device: {config.device}")

    train_dataset = LungSegDataset(config.train_img_dir, config.train_mask_dir, config.img_size, augment=True)
    val_dataset = LungSegDataset(config.val_img_dir, config.val_mask_dir, config.img_size, augment=False)

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    model = UNet(in_channels=1, out_channels=1).to(config.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = torch.nn.BCELoss()

    train(model, train_loader, val_loader, optimizer, criterion, config)

if __name__ == "__main__":
    main()



# visualize.py
import matplotlib.pyplot as plt
import os

def plot_metrics(train_losses, val_dices, val_ious, save_dir="results"):
    os.makedirs(save_dir, exist_ok=True)
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(10, 5))
    plt.plot(epochs, train_losses, 'b-o', label='Train Loss')
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(save_dir, "train_loss.png"))
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(epochs, val_dices, 'g-o', label='Val Dice')
    plt.plot(epochs, val_ious, 'r-o', label='Val IoU')
    plt.title("Validation Metrics")
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.legend()
    plt.savefig(os.path.join(save_dir, "val_metrics.png"))
    plt.close()

    print(f"Saved training curves to {save_dir}/")


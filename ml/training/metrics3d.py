"""Metrics 3D cho segmentation đa lớp: Dice & Hausdorff95 (HD95) per-class.

Cách dùng trong vòng validation:
    metrics = SegMetrics(num_classes=cfg.model.out_channels)
    metrics.reset()
    for batch in val_loader:
        preds = sliding_window_inference(...)     # logits [B, C, H, W, D]
        metrics.update(preds, batch["label"])     # labels [B, 1, H, W, D]
    result = metrics.aggregate()                  # dict: dice/hd95 per-class + mean
"""
import torch
from monai.data import decollate_batch
from monai.metrics import DiceMetric, HausdorffDistanceMetric
from monai.transforms import AsDiscrete, Compose, EnsureType


class SegMetrics:
    """Gom Dice & HD95 cho bài toán segmentation đa lớp.

    Tại sao cần xử lý trước (post-transform)?
      - Model trả về *logits* [B, C, ...]. Metric cần nhãn rời rạc dạng one-hot.
      - pred:  argmax theo chiều kênh -> chỉ số lớp -> one-hot [C, ...]
      - label: {0..C-1} -> one-hot [C, ...]
      - include_background=False: bỏ lớp nền (chiếm ~99% voxel, làm Dice cao giả tạo)
        để đánh giá đúng phần khối u.
    """

    def __init__(self, num_classes: int, include_background: bool = False):
        self.num_classes = num_classes
        self.include_background = include_background

        # reduction="mean_batch": trung bình theo batch, GIỮ chiều lớp -> ra số per-class.
        self.dice = DiceMetric(
            include_background=include_background,
            reduction="mean_batch",
        )
        self.hd95 = HausdorffDistanceMetric(
            include_background=include_background,
            percentile=95,                # HD95: bỏ qua 5% điểm xa nhất -> bớt nhạy outlier
            reduction="mean_batch",
        )

        self.post_pred = Compose(
            [EnsureType(), AsDiscrete(argmax=True, to_onehot=num_classes)]
        )
        self.post_label = Compose(
            [EnsureType(), AsDiscrete(to_onehot=num_classes)]
        )

    @torch.no_grad()
    def update(self, preds: torch.Tensor, labels: torch.Tensor) -> None:
        """preds: logits [B, C, H, W, D]; labels: [B, 1, H, W, D] (chỉ số lớp).

        decollate_batch tách batch thành list từng sample (MONAI metric nhận list).
        """
        preds = [self.post_pred(p) for p in decollate_batch(preds)]
        labels = [self.post_label(lb) for lb in decollate_batch(labels)]
        self.dice(y_pred=preds, y=labels)
        self.hd95(y_pred=preds, y=labels)

    def aggregate(self) -> dict:
        """Tổng hợp cuối epoch. Trả về tensor per-class + trung bình các lớp foreground."""
        dice = self.dice.aggregate()   # shape [n_foreground_classes]
        hd95 = self.hd95.aggregate()
        return {
            "dice": dice,
            "hd95": hd95,
            "mean_dice": float(torch.nanmean(dice)),
            "mean_hd95": float(torch.nanmean(hd95)),
        }

    def reset(self) -> None:
        """Xóa buffer tích lũy — gọi ở ĐẦU mỗi epoch validation."""
        self.dice.reset()
        self.hd95.reset()


if __name__ == "__main__":
    # Smoke test với tensor giả: chỉ cần chạy không lỗi (số ra ngẫu nhiên).
    torch.manual_seed(0)
    m = SegMetrics(num_classes=2)
    m.reset()
    preds = torch.randn(2, 2, 32, 32, 32)                       # logits giả
    labels = torch.randint(0, 2, (2, 1, 32, 32, 32)).float()    # nhãn giả {0,1}
    m.update(preds, labels)
    print(m.aggregate())
    print("OK metrics3d")

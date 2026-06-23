"""Test SegMetrics: dự đoán hoàn hảo -> Dice ~ 1.0; reset xóa buffer."""
import torch

from ml.training.metrics3d import SegMetrics


def _perfect_pred(label):
    """Tạo logits sao cho argmax theo kênh == label (dự đoán hoàn hảo)."""
    c = 2
    logits = torch.zeros(label.shape[0], c, *label.shape[2:])
    for k in range(c):
        logits[:, k] = (label[:, 0] == k).float() * 10.0
    return logits


def test_perfect_prediction_dice_one():
    label = torch.zeros(1, 1, 8, 8, 8)
    label[..., 2:5, 2:5, 2:5] = 1            # một khối u nhỏ
    preds = _perfect_pred(label)

    m = SegMetrics(num_classes=2)
    m.reset()
    m.update(preds, label)
    res = m.aggregate()
    assert res["mean_dice"] > 0.99           # khớp hoàn toàn


def test_reset_no_state_leak():
    """Sau reset, một phép đo mới phải độc lập với phép đo cũ."""
    label = torch.zeros(1, 1, 8, 8, 8)
    label[..., 2:5, 2:5, 2:5] = 1
    good = _perfect_pred(label)
    bad = torch.zeros(1, 2, 8, 8, 8)
    bad[:, 0] = 10.0                         # đoán TOÀN nền -> bỏ sót u

    m = SegMetrics(num_classes=2)
    m.reset(); m.update(bad, label)
    dice_bad = m.aggregate()["mean_dice"]

    m.reset(); m.update(good, label)         # reset rồi đo lại bằng pred hoàn hảo
    dice_good = m.aggregate()["mean_dice"]

    assert dice_good > 0.99                  # không bị kéo xuống bởi lần đo "bad" trước
    assert dice_good > dice_bad

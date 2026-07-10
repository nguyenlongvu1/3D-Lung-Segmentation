"""Test ml.data.download: gọi DecathlonDataset đúng tham số + nuốt lỗi in mẫu thử.

Monkeypatch DecathlonDataset để KHÔNG tải ~9GB dữ liệu thật khi chạy test.
"""
import ml.data.download as dl
from ml.config import load_config


class _FakeDS:
    """Giả lập DecathlonDataset: ghi lại kwargs, trả 1 mẫu để in thử."""

    last_kwargs = None

    def __init__(self, **kwargs):
        _FakeDS.last_kwargs = kwargs

    def __len__(self):
        return 3

    def __getitem__(self, i):
        return {"image": "img.nii.gz", "label": "lbl.nii.gz"}


def test_download_calls_decathlon_with_download_flag(monkeypatch, capsys):
    monkeypatch.setattr(dl, "DecathlonDataset", _FakeDS)
    cfg = load_config()
    dl.download(cfg)
    kwargs = _FakeDS.last_kwargs
    assert kwargs["download"] is True
    assert kwargs["task"] == cfg.data.task
    assert kwargs["section"] == "training"
    out = capsys.readouterr().out
    assert "Số mẫu training: 3" in out


def test_download_survives_sample_print_error(monkeypatch, capsys):
    """Nếu in mẫu thử lỗi, download vẫn báo hoàn tất (không raise)."""

    class _BadItemDS(_FakeDS):
        def __getitem__(self, i):
            raise RuntimeError("boom")

    monkeypatch.setattr(dl, "DecathlonDataset", _BadItemDS)
    dl.download(load_config())
    out = capsys.readouterr().out
    assert "Tải/giải nén hoàn tất" in out


def test_main_parses_config_arg(monkeypatch):
    monkeypatch.setattr(dl, "DecathlonDataset", _FakeDS)
    monkeypatch.setattr("sys.argv", ["download"])
    dl.main()  # không truyền --config -> dùng mặc định, không raise

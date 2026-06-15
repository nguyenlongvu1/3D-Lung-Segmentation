"""Tải dữ liệu Medical Segmentation Decathlon về thư mục data/ và in vài thông tin kiểm tra.

Dùng:
    python -m ml.data.download                       # dùng config mặc định
    python -m ml.data.download --config path.yaml    # config tùy chỉnh
"""
import argparse

from monai.apps import DecathlonDataset

from ml.config import load_config


def download(cfg) -> None:
    print(f"Đang tải/kiểm tra task '{cfg.data.task}' vào '{cfg.data.root}' ...")
    # transform=None -> chỉ tải file, không tiền xử lý; cache_rate=0 để khỏi tốn RAM.
    ds = DecathlonDataset(
        root_dir=cfg.data.root,
        task=cfg.data.task,
        section="training",
        transform=None,
        download=True,
        val_frac=cfg.data.val_frac,
        cache_rate=0.0,
        num_workers=cfg.data.num_workers,
    )
    print(f"Số mẫu training: {len(ds)}")
    # Phần in thử mẫu chỉ để kiểm tra — bọc try/except để không làm hỏng kết quả tải.
    try:
        sample = ds[0]
        print(f"Ví dụ mẫu đầu tiên: image={sample['image']} | label={sample['label']}")
    except Exception as e:  # noqa: BLE001
        print(f"(Tải xong, nhưng không in được mẫu thử: {e})")
    print("✅ Tải/giải nén hoàn tất.")


def main():
    parser = argparse.ArgumentParser(description="Tải dataset MSD cho lung segmentation")
    parser.add_argument("--config", default=None, help="Đường dẫn file YAML cấu hình")
    args = parser.parse_args()
    download(load_config(args.config))


if __name__ == "__main__":
    main()

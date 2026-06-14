---
title: Lung Segmentation App
emoji: 🫁
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
---

> Lung & lesion segmentation on CT — đang được nâng cấp từ U-Net 2D (PNG) thành
> pipeline 3D multi-class trên dữ liệu CT thật (MONAI). Xem lộ trình ở cuối file.

## Project structure

- `api/`: FastAPI backend phục vụ inference (endpoint dự đoán + giao diện web).
- `web/`: frontend assets (templates Jinja2 + static).
- `ml/training/`: baseline U-Net 2D — training, evaluation, prediction trên ảnh PNG.
- `models/`: trọng số model dùng cho inference (`unet_lung_seg.pth`).
- `data/`: dữ liệu (không commit vào git — xem `.gitignore`).

## Docker options

- Root Dockerfile: quick default image for web app inference.
- docker/Dockerfile.app: dedicated app-serving image.
- docker/Dockerfile.train: dedicated training image.

## Dependencies

- requirements.txt: dependencies for inference app and API.
- requirements.train.txt: extra dependencies for training (extends requirements.txt).

Example build commands:

- App image: docker build -f docker/Dockerfile.app -t lung-seg-app .
- Train image: docker build -f docker/Dockerfile.train -t lung-seg-train .

## Roadmap (flagship upgrade)

Đang phát triển theo kế hoạch nâng cấp thành project medical-imaging full-stack:

1. Dữ liệu 3D CT thật (Medical Segmentation Decathlon — Task06 Lung, NIfTI).
2. Pipeline MONAI: transforms 3D, kiến trúc hiện đại (UNet 3D, SwinUNETR).
3. Đánh giá đa chỉ số per-class (Dice, Hausdorff95) + benchmark + tracking (W&B).
4. Demo viewer thể tích (kéo lát cắt CT + overlay mask multi-class).

Baseline U-Net 2D hiện tại được giữ làm mốc đối chứng.
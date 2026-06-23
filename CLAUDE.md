# CLAUDE.md — Lung Tumor Segmentation 3D

Tài liệu ngữ cảnh cho dự án. Đọc file này đầu mỗi phiên để tiếp tục liền mạch.

---

## 1. Dự án là gì & mục tiêu

Nâng cấp một project U-Net 2D segment phổi (tutorial, ảnh PNG) thành **flagship
medical-imaging full-stack** để **xin việc Computer Vision / AI Engineer (fresher)**.

Hướng đã chốt: **đào sâu medical imaging + full-stack**, GPU local (RTX 4060 8GB),
ngân sách 1-2 tháng. Bài toán: **segment khối u phổi trên CT 3D** (Medical Segmentation
Decathlon — Task06_Lung), pipeline **MONAI**, có demo web + định lượng thể tích u.

**Người dùng:** fresher AI Engineer, đang học. Làm việc bằng **tiếng Việt**.
- Ưu tiên **hiểu sâu**, không muốn "AI làm hết". Với code ML lõi (metrics, train loop)
  nên mời họ **tự nháp rồi mình review**; phần boilerplate thì mình viết được.
- Coi trọng **đánh giá trung thực** (held-out test, không khoe số ảo).

---

## 2. Trạng thái hiện tại (mốc cuối phiên trước)

- **Nhánh:** `refactor/monai-scaffold` (CHƯA push). Remote `origin` là **Hugging Face
  Space** (`huggingface.co/spaces/nguyenlongvu1/lung-segmentation`) — push lên `main`
  = auto-deploy. Chưa có remote GitHub.
- **Môi trường đã chạy được:** Python 3.13, torch 2.8.0+cu126 (CUDA OK), MONAI 1.5.2,
  numpy 2.4.6, trong `.venv`.
- **Pipeline hoàn chỉnh & đã train thật.** Model `monai_unet` (UNet 3D, 4.8M params).
- **Kết quả test (held-out 9 ca):** `results/benchmark.csv`
  → **test Dice 0.3644 | HD95 56.72 | 0.45s/vol**. (SOTA nnU-Net task này ~0.65-0.74;
  0.36 là khiêm tốn nhưng trung thực cho UNet nhẹ + 45 ca train + 8GB.)
- **Demo web CHẠY ĐƯỢC:** upload `.nii.gz` → viewer kéo lát cắt + overlay u đỏ +
  **thể tích u (ml)**. Đã test ok (vd lung_035 → 7.58 ml, hướng axial đúng).
- **Checkpoint:** `checkpoints/best_monai_unet.pth` (gitignored).

### ⚠️ Có thay đổi CHƯA COMMIT (cần commit để khỏi mất):
- `api/inference.py` (mới), `api/main.py`, `web/templates/index.html`, `web/static/styles.css`
  → **toàn bộ demo web**.
- `ml/training/train3d.py`, `ml/data/transforms.py`, `ml/config/default.yaml`
  → loss config + aug/scheduler thành tùy chọn (mặc định tắt).
- `README.md`, `CLAUDE.md`.

---

## 3. Cấu trúc thư mục

```
ml/
  config/default.yaml      # TẤT CẢ siêu tham số. __init__.py: load_config/save_config (dot-access)
  data/
    download.py            # tải MSD Task06_Lung (~9GB) qua DecathlonDataset (AWS mirror, public)
    transforms.py          # MONAI transforms 3D (HU windowing, spacing, patch, augment)
    datamodule.py          # split train/val/test 3 phần (seed) + get_loaders/get_test_loader
    check.py               # sanity-check pipeline -> results/sanity_check.png
  models/
    factory.py             # build_model: monai_unet | swin_unetr + count_parameters
  training/
    train3d.py             # TRAIN LOOP CHÍNH (DiceFocalLoss, AMP, sliding-window, early stop)
    evaluate3d.py          # đánh giá trên TEST -> ghi results/benchmark.csv
    metrics3d.py           # SegMetrics: Dice + HD95 per-class (include_background=False)
api/
  inference.py             # LungTumorInference: load model + tiền xử lý + sliding-window + thể tích u
  main.py                  # FastAPI: / , POST /predict (.nii.gz), GET /slice/{z} (PNG overlay)
web/templates/index.html   # viewer: upload + slider lát cắt + overlay + badge thể tích
web/static/styles.css
docker/Dockerfile.app, Dockerfile.train ; Dockerfile (root, app)
requirements.txt           # inference (fastapi, torch, monai, nibabel...)
requirements.train.txt     # +train (wandb, tensorboard, scikit-image...) + extra-index cu126
data/                      # gitignored. data/Task06_Lung/{imagesTr,labelsTr,imagesTs}/*.nii.gz
checkpoints/, results/, wandb/  # gitignored
```

---

## 4. Môi trường — CÁCH CÀI (Python 3.13, nhiều cạm bẫy đã giải)

```bash
# 1. torch bản CUDA (PHẢI cài riêng, không để requirements kéo bản CPU)
pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu126
# 2. phần còn lại
pip install -r requirements.train.txt
# 3. kiểm tra
python -c "import torch, monai, numpy; print(torch.__version__, torch.cuda.is_available(), monai.__version__, numpy.__version__)"
# kỳ vọng: 2.8.0+cu126 True 1.5.2 2.4.6
```

**Cạm bẫy dependency đã xử lý (đừng lặp lại):**
- Python 3.13 **bắt buộc numpy ≥ 2.1** (không có wheel chính thức cho numpy 1.26 → bản
  MinGW lỗi "CRASHES ARE TO BE EXPECTED"). → dùng numpy 2.x.
- MONAI **1.4.0 cấm numpy<2.0** → KHÔNG hợp Python 3.13. Phải **MONAI 1.5.2** (hỗ trợ numpy 2).
- `opencv-python`/`albumentations` (bản py3.13) đòi numpy≥2 nhưng mâu thuẫn cấu hình cũ →
  **tách khỏi requirements.train.txt** (chỉ baseline 2D cần, cài riêng nếu muốn).
- `--extra-index-url .../cu126` đặt đầu requirements.train.txt để torch/torchvision khớp bản GPU.

---

## 5. Các lệnh chính

```bash
python -m ml.data.download                         # tải data (1 lần)
python -m ml.data.check                            # sanity-check pipeline
python -m ml.training.train3d                      # train (monai_unet, config mặc định)
python -m ml.training.train3d --model swin_unetr --batch-size 1   # đối chứng (8GB cần batch 1)
python -m ml.training.train3d --epochs 2           # smoke test nhanh
python -m ml.training.evaluate3d                   # eval test -> benchmark.csv
uvicorn api.main:app --reload --port 7860          # DEMO -> http://127.0.0.1:7860
```

Demo: chọn file thật trong `data/Task06_Lung/imagesTr/` (vd `lung_001.nii.gz`).
**Tránh file `._*`** (rác AppleDouble của macOS — LoadImage sẽ lỗi).

---

## 6. Config quan trọng (`ml/config/default.yaml`)

- `data`: `task=Task06_Lung`, `spacing=[1.5,1.5,2.0]`, `a_min=-1000/a_max=400` (HU phổi),
  `roi_size=[96,96,96]`, `num_samples=4`, `batch_size=2`, `val_frac=0.15`, `test_frac=0.15`,
  `cache_rate=1.0`, `num_workers=4`, **`loader_workers=0`** (Windows!), `strong_aug=false`.
- `model`: `name=monai_unet`, `in_channels=1`, `out_channels=2` (nền + u).
- `train`: `max_epochs=300`, **`loss=dicefocal`**, `lr=0.0001`, **`use_scheduler=false`**,
  `val_interval=2`, `early_stop_patience=15`, **`early_stop_min_dice=0.05`**, `amp=true`,
  `sw_batch_size=4`, `sw_overlap=0.5`.

---

## 7. Bài học & quyết định quan trọng (ĐỪNG lặp sai lầm)

1. **Foreground collapse (u quá nhỏ ~0.1% voxel):** `DiceCELoss` mặc định → model đoán
   TOÀN NỀN, loss giảm nhưng Dice→0. → đổi **`DiceFocalLoss`**. Cũng có `TverskyLoss` làm dự phòng.
2. **"Collapse-then-recover":** Dice ≈0 trong ~38-40 epoch ĐẦU rồi MỚI bật lên (đã thấy
   leo tới val 0.62). → **KIÊN NHẪN, đừng dừng sớm.** (Đã lỡ bảo dừng 1 lần — sai.)
3. **Early stopping phải bỏ qua pha collapse:** thêm `early_stop_min_dice=0.05` — chỉ đếm
   patience SAU khi best dice vượt ngưỡng. Nếu không, nó dừng oan ngay trước lúc hồi phục.
4. **Aug mạnh + LR scheduler (cosine) → HẠI trên data nhỏ này** (collapse sâu hơn, khó thoát).
   → đã revert thành tùy chọn `strong_aug=false` / `use_scheduler=false`. **Phức tạp ≠ tốt hơn.**
5. **Windows + CacheDataset:** `DataLoader num_workers>0` nhân bản cache → tràn RAM.
   → `loader_workers=0`. (`num_workers` chỉ dùng cho cache ban đầu.)
6. **Gap val (0.62) vs test (0.36):** val-best bị thiên lệch lạc quan + set 9 ca nhiễu.
   Tập **test held-out mới là số trung thực**. Đây là điểm nhấn cho báo cáo.
7. **Model nặng (SwinUNETR) khó deploy hơn** (VRAM, tốc độ, image, chi phí). Benchmark có
   cột params/infer_time để lập luận đánh đổi accuracy-vs-deployability.

---

## 8. Roadmap & việc còn lại

**Đã xong:** G0 dọn repo · G1 pipeline MONAI · G2 train monai_unet · G3 metrics+eval+benchmark ·
G4 demo web (+ định lượng thể tích).

**Còn lại (ưu tiên giảm dần):**
1. **Commit** toàn bộ thay đổi đang treo (demo + experiments + CLAUDE.md).
2. **README hoàn chỉnh:** bài toán, dữ liệu, phương pháp, **bảng kết quả** (test Dice 0.36,
   HD95, params, tốc độ), **ảnh demo** (screenshot có overlay), hạn chế, hướng dẫn chạy.
3. (tùy chọn) **SwinUNETR** đối chứng → benchmark 2 dòng (`--model swin_unetr --batch-size 1`,
   factory đã bật gradient checkpointing cho 8GB).
4. (tùy chọn) **K-fold CV** → con số vững thay vì 1 phép đo nhiễu 9 ca (chuẩn vàng data nhỏ).
5. (tùy chọn) MLOps: export ONNX, docker-compose, vài pytest, GitHub Action CI.
6. (tùy chọn) Tạo **repo GitHub** làm nơi chính cho portfolio (recruiter xem GitHub).
7. (tùy chọn) Deploy HF Space: cần xử lý ship checkpoint (`.pth` đang gitignored) — Git LFS
   hoặc release artifact; requirements.txt nặng hơn (có MONAI).

---

## 9. Quy ước

- **Commit message:** tiếng Việt, prefix `feat/fix/refactor`, kết thúc bằng:
  `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`
- Không code/đẩy thẳng lên `main` (HF). Làm trên nhánh `refactor/monai-scaffold`.
- `.gitignore` dùng `/data/` (neo gốc) để KHÔNG nuốt nhầm package `ml/data/`.
- Mọi thí nghiệm nên đi qua config (loss/strong_aug/use_scheduler đã configurable).

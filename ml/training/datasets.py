import os
import cv2
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

class LungSegDataset(Dataset):
    def __init__(self, img_dir, mask_dir, img_size=256, augment=True):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.augment = augment

        # Ghép ảnh và mask theo tên file chung (không dựa vào thứ tự sort độc lập,
        # tránh ghép sai khi hai thư mục lệch nhau).
        img_files = set(os.listdir(img_dir))
        mask_files = set(os.listdir(mask_dir))
        self.ids = sorted(img_files & mask_files)

        if not self.ids:
            raise RuntimeError(
                f"Không tìm thấy cặp ảnh-mask trùng tên giữa {img_dir} và {mask_dir}"
            )
        missing = (img_files ^ mask_files)
        if missing:
            print(f"[LungSegDataset] Cảnh báo: {len(missing)} file chỉ có ở một phía, đã bỏ qua.")

        # --- augmentation pipeline ---
        if augment:
            self.transform = A.Compose([
                A.Resize(img_size, img_size),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=15, p=0.5),
                A.RandomBrightnessContrast(p=0.5),
                A.GaussianBlur(blur_limit=(3, 5), p=0.3),
                ToTensorV2(),
            ])
        else:
            # chỉ resize và chuyển sang tensor
            self.transform = A.Compose([
                A.Resize(img_size, img_size),
                ToTensorV2(),
            ])

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        name = self.ids[idx]
        img_path = os.path.join(self.img_dir, name)
        mask_path = os.path.join(self.mask_dir, name)

        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if image is None or mask is None:
            raise FileNotFoundError(f"Không đọc được ảnh/mask cho '{name}'")

        # Apply same transformation
        augmented = self.transform(image=image, mask=mask)
        image = augmented["image"]
        mask = augmented["mask"]

        # Ensure both are float tensors of shape [1, H, W]
        image = image.float() / 255.0
        mask = mask.float() / 255.0

        return image, mask.unsqueeze(0)


import os
import torch
import torchvision
import numpy as np
from model import UNET
from utils import get_loaders
from PIL import Image
from albumentations.pytorch import ToTensorV2
import albumentations as A

# --- Config ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = "my_checkpoint.pth.tar"
SAVE_DIR = "test_output"
IMG_DIR = "data/val_images/"
MASK_DIR = "data/val_masks/"
BATCH_SIZE = 1
IMG_HEIGHT = 160
IMG_WIDTH = 240

os.makedirs(SAVE_DIR, exist_ok=True)

# --- Transforms ---
val_transform = A.Compose([
    A.Resize(height=IMG_HEIGHT, width=IMG_WIDTH),
    A.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0], max_pixel_value=255.0),
    ToTensorV2(),
])

# --- Load model ---
model = UNET(in_channels=3, out_channels=1).to(DEVICE)
checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
model.load_state_dict(checkpoint["state_dict"])
model.eval()

# --- Get data loader ---
_, val_loader = get_loaders(
    train_dir=None, train_maskdir=None,
    val_dir=IMG_DIR, val_maskdir=MASK_DIR,
    batch_size=BATCH_SIZE, train_transform=None,
    val_transform=val_transform, num_workers=0, pin_memory=False
)

# --- Run inference ---
print("🧪 Running inference on validation set...")
with torch.no_grad():
    for idx, (x, y) in enumerate(val_loader):
        x = x.to(DEVICE)
        preds = torch.sigmoid(model(x))
        preds = (preds > 0.5).float()

        # Save overlay
        pred_np = preds.cpu().squeeze().numpy() * 255.0
        img_tensor = x.cpu().squeeze().permute(1, 2, 0).numpy() * 255.0

        overlay = np.stack([img_tensor[..., 0], img_tensor[..., 1], img_tensor[..., 2]], axis=-1)
        pred_mask = np.uint8(pred_np)
        overlay[..., 1] = np.maximum(overlay[..., 1], pred_mask)  # Green for mask

        save_path = os.path.join(SAVE_DIR, f"pred_{idx}.png")
        Image.fromarray(np.uint8(overlay)).save(save_path)
        print(f"[✓] Saved: {save_path}")

print("✅ All predictions saved.")

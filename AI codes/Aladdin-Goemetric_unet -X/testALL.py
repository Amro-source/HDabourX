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

# --- IoU Calculation ---
def calculate_iou(y_true, y_pred):
    intersection = np.logical_and(y_true, y_pred)
    union = np.logical_or(y_true, y_pred)
    return np.sum(intersection) / np.sum(union) if np.sum(union) != 0 else 0.0

# --- Run inference ---
print("🧪 Running inference on validation set...")
iou_scores = []
report_lines = []

with torch.no_grad():
    for idx, (x, y) in enumerate(val_loader):
        x = x.to(DEVICE)
        y = y.to(DEVICE).float()
        preds = torch.sigmoid(model(x))
        preds = (preds > 0.5).float()

        pred_np = preds.cpu().squeeze().numpy()
        target_np = y.cpu().squeeze().numpy()
        input_np = x.cpu().squeeze().permute(1, 2, 0).numpy() * 255.0

        iou = calculate_iou(target_np, pred_np)
        iou_scores.append(iou)
        report_lines.append(f"Image {idx}: IoU = {iou:.4f}")

        # Save original image
        input_img = Image.fromarray(np.uint8(input_np))
        input_img.save(os.path.join(SAVE_DIR, f"image_{idx}.png"))

        # Save predicted mask
        pred_mask = Image.fromarray((pred_np * 255).astype(np.uint8))
        pred_mask.save(os.path.join(SAVE_DIR, f"pred_mask_{idx}.png"))

        # Save ground truth mask
        gt_mask = Image.fromarray((target_np * 255).astype(np.uint8))
        gt_mask.save(os.path.join(SAVE_DIR, f"gt_mask_{idx}.png"))

        print(f"[✓] Saved image, prediction, and mask for index {idx} | IoU: {iou:.4f}")

# --- Save report ---
mean_iou = np.mean(iou_scores)
std_iou = np.std(iou_scores)
min_iou = np.min(iou_scores)
max_iou = np.max(iou_scores)

report_lines.append("\n--- Summary ---")
report_lines.append(f"Mean IoU: {mean_iou:.4f}")
report_lines.append(f"Std Dev IoU: {std_iou:.4f}")
report_lines.append(f"Min IoU: {min_iou:.4f}")
report_lines.append(f"Max IoU: {max_iou:.4f}")

report_path = os.path.join(SAVE_DIR, "report.txt")
with open(report_path, "w") as f:
    for line in report_lines:
        f.write(line + "\n")

print(f"✅ All predictions and masks saved. Report written to: {report_path}")

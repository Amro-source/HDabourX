import cv2
import numpy as np
import os
import random
from tqdm import tqdm

# Setup directories
data_dir = "unet_multimask_dataset"
os.makedirs(os.path.join(data_dir, "images"), exist_ok=True)
os.makedirs(os.path.join(data_dir, "masks"), exist_ok=True)

# Parameters
IMG_SIZE = 192
NUM_TRAIN = 200  # Reduced for testing - change to 2000 for full dataset
NUM_VAL = 20     # Reduced for testing - change to 200 for full dataset
SHAPE_TYPES = ["circle", "square", "rectangle", "ellipse"]
SHAPE_COLORS = {
    "circle": (255, 0, 0),     # Red
    "square": (0, 255, 0),     # Green
    "rectangle": (0, 0, 255),  # Blue
    "ellipse": (0, 255, 255)   # Yellow
}

def generate_image_and_masks():
    """Generate image with shapes and separate masks for each shape type"""
    img = np.ones((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8) * 255  # White background
    masks = {shape: np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8) for shape in SHAPE_TYPES}
    occupied = []
    
    # Generate 3-5 shapes
    for _ in range(random.randint(3, 5)):
        shape = random.choice(SHAPE_TYPES)
        placed = False
        
        for attempt in range(100):  # Try 100 times to place each shape
            if shape == "circle":
                radius = random.randint(20, 50)
                center = (random.randint(radius, IMG_SIZE-radius), 
                         random.randint(radius, IMG_SIZE-radius))
                bbox = (center[0]-radius, center[1]-radius, 
                        center[0]+radius, center[1]+radius)
            elif shape == "square":
                side = random.randint(30, 60)
                x1 = random.randint(0, IMG_SIZE-side)
                y1 = random.randint(0, IMG_SIZE-side)
                bbox = (x1, y1, x1+side, y1+side)
            elif shape == "rectangle":
                w, h = random.randint(30, 60), random.randint(30, 60)
                x1 = random.randint(0, IMG_SIZE-w)
                y1 = random.randint(0, IMG_SIZE-h)
                bbox = (x1, y1, x1+w, y1+h)
            elif shape == "ellipse":
                axes = (random.randint(20, 40), random.randint(20, 40))
                center = (random.randint(axes[0], IMG_SIZE-axes[0]), 
                         random.randint(axes[1], IMG_SIZE-axes[1]))
                angle = random.randint(0, 360)
                bbox = (center[0]-axes[0], center[1]-axes[1],
                        center[0]+axes[0], center[1]+axes[1])
            
            # Check for overlaps
            overlap = any((bbox[0] < existing[2] and bbox[2] > existing[0] and
                          bbox[1] < existing[3] and bbox[3] > existing[1])
                          for existing in occupied)
            
            if not overlap:
                # Draw colored shape on main image
                color = SHAPE_COLORS[shape]
                if shape == "circle":
                    cv2.circle(img, center, radius, color, -1)
                    cv2.circle(masks[shape], center, radius, 255, -1)
                elif shape in ["square", "rectangle"]:
                    cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, -1)
                    cv2.rectangle(masks[shape], (bbox[0], bbox[1]), (bbox[2], bbox[3]), 255, -1)
                elif shape == "ellipse":
                    cv2.ellipse(img, center, axes, angle, 0, 360, color, -1)
                    cv2.ellipse(masks[shape], center, axes, angle, 0, 360, 255, -1)
                
                occupied.append(bbox)
                placed = True
                break
        
        if not placed:
            print(f"Warning: Couldn't place {shape} after 100 attempts")
    
    return img, masks

# Generate dataset
print("Creating training set...")
for i in tqdm(range(NUM_TRAIN)):
    img, masks = generate_image_and_masks()
    cv2.imwrite(os.path.join(data_dir, "images", f"train_{i:04d}.png"), img)
    for shape, mask in masks.items():
        cv2.imwrite(os.path.join(data_dir, "masks", f"train_{i:04d}_{shape}.png"), mask)

print("\nCreating validation set...")
for i in tqdm(range(NUM_VAL)):
    img, masks = generate_image_and_masks()
    cv2.imwrite(os.path.join(data_dir, "images", f"val_{i:04d}.png"), img)
    for shape, mask in masks.items():
        cv2.imwrite(os.path.join(data_dir, "masks", f"val_{i:04d}_{shape}.png"), mask)

# Verification
sample_files = [f for f in os.listdir(os.path.join(data_dir, "masks")) if f.startswith("train_0000")]
print("\nGenerated mask files for first training sample:")
for f in sample_files:
    print(f"- {f}")

print("\nDataset structure:")
print(f"images/train_XXXX.png - Main RGB images")
print(f"masks/train_XXXX_circle.png - Circle masks (white on black)")
print(f"masks/train_XXXX_square.png - Square masks")
print(f"masks/train_XXXX_rectangle.png - Rectangle masks")
print(f"masks/train_XXXX_ellipse.png - Ellipse masks")
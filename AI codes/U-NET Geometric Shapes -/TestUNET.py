import os
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from keras.models import load_model

# Load your trained model
model = load_model('shape_segmentation_model.hdf5')

# Directories
data_dir = "unet_multimask_dataset"
test_image_dir = os.path.join(data_dir, "images")
test_mask_dir = os.path.join(data_dir, "masks")

# Get test images (using the first 5 for demonstration)
test_images = sorted([f for f in os.listdir(test_image_dir) 
                     if f.endswith('.png') and not any(x in f for x in ['circle', 'square', 'rectangle', 'ellipse'])])[:5]

def preprocess_image(img_path, size=256):
    """Load and preprocess a single image"""
    img = cv2.imread(img_path, 0)  # Read as grayscale
    img = Image.fromarray(img).resize((size, size))
    img_array = np.array(img, dtype='float32') / 255.0
    return np.expand_dims(np.expand_dims(img_array, axis=-1), axis=0)

def visualize_predictions(image_files, num_samples=3):
    """Visualize model predictions on sample images"""
    plt.figure(figsize=(15, 5*num_samples))
    
    for i, img_file in enumerate(image_files[:num_samples]):
        # Load and preprocess image
        img_path = os.path.join(test_image_dir, img_file)
        img = cv2.imread(img_path, 0)
        img_display = Image.fromarray(img).resize((256, 256))
        
        # Get corresponding masks
        base_name = img_file.split('.')[0]
        mask_files = [f for f in os.listdir(test_mask_dir) if f.startswith(base_name)]
        
        # Create ground truth combined mask
        gt_mask = np.zeros((256, 256), dtype='float32')
        for mask_file in mask_files:
            mask_path = os.path.join(test_mask_dir, mask_file)
            mask = cv2.imread(mask_path, 0)
            mask = Image.fromarray(mask).resize((256, 256))
            gt_mask = np.maximum(gt_mask, np.array(mask, dtype='float32')/255.0)
        
        # Model prediction
        processed_img = preprocess_image(img_path)
        pred_mask = model.predict(processed_img)[0,:,:,0]
        pred_mask_thresh = (pred_mask > 0.5).astype(np.uint8)
        
        # Plot results
        plt.subplot(num_samples, 3, i*3 + 1)
        plt.imshow(img_display, cmap='gray')
        plt.title(f"Input Image: {img_file}")
        plt.axis('off')
        
        plt.subplot(num_samples, 3, i*3 + 2)
        plt.imshow(gt_mask, cmap='gray')
        plt.title("Ground Truth Mask")
        plt.axis('off')
        
        plt.subplot(num_samples, 3, i*3 + 3)
        plt.imshow(pred_mask_thresh, cmap='gray')
        plt.title("Predicted Mask")
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()

# Run visualization
visualize_predictions(test_images, num_samples=3)

# Additional metrics calculation
def calculate_iou(y_true, y_pred):
    intersection = np.logical_and(y_true, y_pred)
    union = np.logical_or(y_true, y_pred)
    return np.sum(intersection) / np.sum(union)

# Evaluate on all test images
print("\nEvaluating on test images...")
iou_scores = []
for img_file in test_images:
    img_path = os.path.join(test_image_dir, img_file)
    base_name = img_file.split('.')[0]
    
    # Load and process image
    processed_img = preprocess_image(img_path)
    pred_mask = (model.predict(processed_img)[0,:,:,0] > 0.5).astype(np.uint8)
    
    # Create ground truth mask
    mask_files = [f for f in os.listdir(test_mask_dir) if f.startswith(base_name)]
    gt_mask = np.zeros((256, 256), dtype=np.uint8)
    for mask_file in mask_files:
        mask_path = os.path.join(test_mask_dir, mask_file)
        mask = cv2.imread(mask_path, 0)
        mask = Image.fromarray(mask).resize((256, 256))
        gt_mask = np.maximum(gt_mask, (np.array(mask)/255.0 > 0.5).astype(np.uint8))
    
    # Calculate IoU
    iou = calculate_iou(gt_mask, pred_mask)
    iou_scores.append(iou)
    print(f"{img_file}: IoU = {iou:.4f}")

print(f"\nAverage IoU: {np.mean(iou_scores):.4f}")
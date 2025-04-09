import os
import cv2
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from simple_unet_model import simple_unet_model
from sklearn.model_selection import train_test_split
from keras.optimizers import Adam

# Data directories
data_dir = "unet_multimask_dataset"
image_directory = os.path.join(data_dir, "images")
mask_directory = os.path.join(data_dir, "masks")

SIZE = 256

def load_data(image_dir, mask_dir, size=SIZE):
    """Load and preprocess images and masks"""
    image_dataset = []
    mask_dataset = []
    
    image_files = sorted([f for f in os.listdir(image_dir) 
                         if f.endswith('.png') and not any(x in f for x in ['circle', 'square', 'rectangle', 'ellipse'])])
    mask_files = sorted([f for f in os.listdir(mask_dir) if f.endswith('.png')])
    
    print(f"Found {len(image_files)} images and {len(mask_files)} mask files")
    
    mask_groups = {}
    for mask_file in mask_files:
        parts = mask_file.split('_')
        base_name = f"{parts[0]}_{parts[1]}.png"
        shape_type = parts[2].split('.')[0]
        if base_name not in mask_groups:
            mask_groups[base_name] = {}
        mask_groups[base_name][shape_type] = mask_file
    
    for img_file in image_files:
        if img_file in mask_groups:
            try:
                # Load image
                img = cv2.imread(os.path.join(image_dir, img_file), 0)
                img = Image.fromarray(img).resize((size, size))
                img_array = np.array(img, dtype='float32') / 255.0
                
                # Create combined mask
                combined_mask = np.zeros((size, size), dtype='float32')
                for mask_file in mask_groups[img_file].values():
                    mask = cv2.imread(os.path.join(mask_dir, mask_file), 0)
                    mask = Image.fromarray(mask).resize((size, size))
                    mask_array = np.array(mask, dtype='float32') / 255.0
                    combined_mask = np.maximum(combined_mask, mask_array)
                
                image_dataset.append(img_array)
                mask_dataset.append(combined_mask)
                
            except Exception as e:
                print(f"Error processing {img_file}: {str(e)}")
    
    image_dataset = np.expand_dims(np.array(image_dataset), axis=-1)
    mask_dataset = np.expand_dims(np.array(mask_dataset), axis=-1)
    
    print(f"Successfully loaded {len(image_dataset)} pairs")
    print(f"Final shapes - Images: {image_dataset.shape}, Masks: {mask_dataset.shape}")
    
    return image_dataset, mask_dataset

# Load data
X, y = load_data(image_directory, mask_directory)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=0)

# Build and compile model
def get_model():
    model = simple_unet_model(X.shape[1], X.shape[2], X.shape[3])
    model.compile(optimizer=Adam(learning_rate=1e-3),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

model = get_model()

# Train model
history = model.fit(X_train, y_train,
                    batch_size=16,
                    epochs=30,
                    validation_data=(X_test, y_test))

# Save model
model.save('shape_segmentation_model.hdf5')

# Evaluate
_, acc = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {acc*100:.2f}%")

# Plot training history
plt.figure(figsize=(12, 4))
plt.subplot(121)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.legend()
plt.subplot(122)
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.legend()
plt.show()
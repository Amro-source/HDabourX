import os
import torch
import numpy as np
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import streamlit as st
from model import UNET

# --- Config ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = "my_checkpoint.pth.tar"
IMG_HEIGHT = 160
IMG_WIDTH = 240

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

# --- Set black background using custom CSS ---
def set_background_color():
    st.markdown(
        """
        <style>
        .stApp {
            background-color: black;
            color: white; /* Change text color to white for better visibility */
        }
        h1, h2, h3, h4, h5, h6 {
            color: white !important;
        }
        .css-1d391kg {
            color: white !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

set_background_color()

# --- Streamlit App ---
st.title("UNET Image Segmentation with Black Background")

# File uploader
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess the image
    image_np = np.array(image)
    transformed = val_transform(image=image_np)
    input_tensor = transformed["image"].unsqueeze(0).to(DEVICE)

    # Perform inference
    with torch.no_grad():
        preds = torch.sigmoid(model(input_tensor))
        preds = (preds > 0.5).float()

    # Convert predictions to numpy
    pred_np = preds.cpu().squeeze().numpy() * 255.0
    pred_mask = np.uint8(pred_np)

    # Resize the mask to match the original image size
    pred_mask_resized = np.array(Image.fromarray(pred_mask).resize((image.width, image.height)))

    # Overlay the mask on the original image
    overlay = np.array(image).copy()
    overlay[:, :, 1] = np.maximum(overlay[:, :, 1], pred_mask_resized)  # Green for mask

    # Display the predicted mask and overlay
    st.subheader("Predicted Mask")
    st.image(pred_mask_resized, caption="Predicted Mask", use_column_width=True)

    st.subheader("Overlay")
    st.image(overlay, caption="Mask Overlay on Original Image", use_column_width=True)
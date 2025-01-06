# -*- coding: utf-8 -*-
"""
Created on Mon Dec 30 00:50:25 2024

@author: M5
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import easyocr
# Your EasyOCR code here
import easyocr
import cv2
import matplotlib.pyplot as plt

# Create an EasyOCR Reader
reader = easyocr.Reader(['en'])  # You can specify the languages you want to use

# Load an image using OpenCV
image_path = '1.png'  # Replace with your image path
image = cv2.imread(image_path)

# Perform OCR on the image
results = reader.readtext(image)

# Print the results
for (bbox, text, prob) in results:
    print(f'Detected text: {text} (Confidence: {prob:.2f})')

    # Draw bounding box around detected text
    (top_left, top_right, bottom_right, bottom_left) = bbox
    top_left = tuple(map(int, top_left))
    bottom_right = tuple(map(int, bottom_right))
    cv2.rectangle(image, top_left, bottom_right, (0, 255, 0), 2)

# Display the image with detected text
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.axis('off')  # Hide axes
plt.show()
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 14 23:41:05 2024

@author: Meshmesh
"""

import cv2
import pytesseract
import pandas as pd

# Input image
image_path = "1.png"

# Read the image
image = cv2.imread(image_path)

# Run Tesseract with TSV output
tsv_data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DATAFRAME)

# Filter lines (level 4 corresponds to lines)
lines = tsv_data[tsv_data['level'] == 4]

# Iterate through each line and save cropped images
for i, line in lines.iterrows():
    x, y, w, h = line['left'], line['top'], line['width'], line['height']
    line_image = image[y:y+h, x:x+w]
    
    # Save each line as a separate image
    cv2.imwrite(f"line_{i}.png", line_image)
    print(f"Saved line {i}: {line['text']}")

# Optional: Display lines
for i, line in lines.iterrows():
    print(f"Line {i}: {line['text']}")

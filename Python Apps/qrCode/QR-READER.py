# -*- coding: utf-8 -*-
"""
Created on Fri Dec 13 21:08:39 2024

@author: Meshmesh
"""
import cv2
from pyzbar.pyzbar import decode

# Function to read QR code
def read_qr_code(image_path):
    # Load the image
    image = cv2.imread(image_path)

    # Decode the QR code
    qr_codes = decode(image)

    # Extract and print data from the QR code(s)
    for qr_code in qr_codes:
        print("QR Code Data:", qr_code.data.decode('utf-8'))
        print("QR Code Type:", qr_code.type)

    if not qr_codes:
        print("No QR code found.")

# Path to the QR code image
image_path = "ExampleCode.png"

image_path="QR_code_for_mobile_English_Wikipedia.svg.png"

# Call the function
read_qr_code(image_path)


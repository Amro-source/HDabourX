# -*- coding: utf-8 -*-
"""
Created on Fri Dec 13 21:30:53 2024

@author: Meshmesh
"""

import qrcode

# Data to encode in the QR code
data = "https://example.com"

data="Amr is a cool software engineer"

# Create QR code instance
qr = qrcode.QRCode(
    version=1,  # Version 1 = 21x21 matrix; higher version = more capacity
    error_correction=qrcode.constants.ERROR_CORRECT_L,  # Error correction level
    box_size=10,  # Size of each box in the QR code grid
    border=4,  # Border size (minimum is 4)
)

# Add data to the QR code
qr.add_data(data)
qr.make(fit=True)

# Generate the QR code as an image
img = qr.make_image(fill_color="black", back_color="white")

# Save the image
img.save("qrcodeAmr.png")

print("QR Code generated and saved as qrcode.png")

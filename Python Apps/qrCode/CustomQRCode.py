# -*- coding: utf-8 -*-
"""
Created on Fri Dec 13 21:38:35 2024

@author: Meshmesh
"""

import qrcode

data = "https://example.com"

# Generate QR code with custom colors
qr = qrcode.QRCode(
    version=2,  # Version 2 = larger capacity
    error_correction=qrcode.constants.ERROR_CORRECT_H,  # High error correction
    box_size=10,
    border=4,
)

qr.add_data(data)
qr.make(fit=True)

# Custom color QR code
img = qr.make_image(fill_color="blue", back_color="yellow")
img.save("custom_qrcode.png")

print("Custom QR Code generated and saved as custom_qrcode.png")


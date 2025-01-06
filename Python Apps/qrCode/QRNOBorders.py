# -*- coding: utf-8 -*-
"""
Created on Fri Dec 13 23:24:14 2024

@author: Meshmesh
"""

import qrcode

data = "https://example.com"
qr = qrcode.QRCode(
    version=1,
    error_correction=qrcode.constants.ERROR_CORRECT_L,
    box_size=10,
    border=0,  # No border
)
qr.add_data(data)
qr.make(fit=True)

img = qr.make_image(fill_color="black", back_color="white")
img.save("noborder_qrcode.png")

print("QR Code without border saved as noborder_qrcode.png")

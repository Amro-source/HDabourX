# -*- coding: utf-8 -*-
"""
Created on Fri Dec 13 23:36:31 2024

@author: Meshmesh
"""

import qrcode
import matplotlib.pyplot as plt

data = "https://example.com"

# Generate QR code
img = qrcode.make(data)

# Display the QR code
plt.imshow(img, cmap="gray")
plt.axis("off")
plt.show()

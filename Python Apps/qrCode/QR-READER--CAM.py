# -*- coding: utf-8 -*-
"""
Created on Fri Dec 13 21:22:23 2024

@author: Meshmesh
"""

import cv2
from pyzbar.pyzbar import decode

# Open camera
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Decode the QR code from the frame
    qr_codes = decode(frame)

    for qr_code in qr_codes:
        data = qr_code.data.decode('utf-8')
        print("QR Code Data:", data)
        # Draw a rectangle around the QR code
        points = qr_code.polygon
        if len(points) > 4:
            hull = cv2.convexHull(points)
            points = hull
        points = [(p.x, p.y) for p in points]
        for i in range(len(points)):
            pt1 = points[i]
            pt2 = points[(i + 1) % len(points)]
            cv2.line(frame, pt1, pt2, (0, 255, 0), 3)

    # Display the frame
    cv2.imshow("QR Code Scanner", frame)

    # Exit with 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

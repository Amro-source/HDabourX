import cv2

# Function to decode QR code using OpenCV
def decode_qr_code(image_path):
    # Initialize OpenCV QRCodeDetector
    detector = cv2.QRCodeDetector()

    # Read the image
    image = cv2.imread(image_path)

    # Detect and decode the QR code
    data, vertices, _ = detector.detectAndDecode(image)

    if data:
        print("QR Code Data:", data)
        
        # Draw the bounding box if vertices are detected
        if vertices is not None:
            vertices = vertices[0]  # Extract the vertices array
            for i in range(len(vertices)):
                pt1 = tuple(map(int, vertices[i]))  # Convert to integer tuple
                pt2 = tuple(map(int, vertices[(i + 1) % len(vertices)]))  # Convert to integer tuple
                cv2.line(image, pt1, pt2, (0, 255, 0), 3)

        # Display the image with the bounding box
        cv2.imshow("QR Code", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("No QR code detected in the image.")

# Path to the QR code image
image_path = "qrcode.png"
image_path = "qrcodeAmr.png"


# Decode the QR code
decode_qr_code(image_path)

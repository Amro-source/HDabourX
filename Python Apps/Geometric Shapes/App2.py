import cv2
import numpy as np
import os

# Create the output folder if it doesn't exist
output_folder = "output"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Image dimensions
width, height = 500, 500

# Generate 50 images
for i in range(50):
    # Create a blank image (white background)
    image = np.ones((height, width, 3), dtype=np.uint8) * 255  # White background

    # Draw random shapes
    # Random rectangle
    pt1 = (np.random.randint(0, width // 2), np.random.randint(0, height // 2))
    pt2 = (np.random.randint(width // 2, width), np.random.randint(height // 2, height))
    color = (np.random.randint(0, 256), np.random.randint(0, 256), np.random.randint(0, 256))  # Random color
    cv2.rectangle(image, pt1, pt2, color, -1)  # Filled rectangle

    # Random circle
    center = (np.random.randint(0, width), np.random.randint(0, height))
    radius = np.random.randint(10, 100)
    color = (np.random.randint(0, 256), np.random.randint(0, 256), np.random.randint(0, 256))  # Random color
    cv2.circle(image, center, radius, color, -1)  # Filled circle

    # Random line
    pt1 = (np.random.randint(0, width), np.random.randint(0, height))
    pt2 = (np.random.randint(0, width), np.random.randint(0, height))
    color = (np.random.randint(0, 256), np.random.randint(0, 256), np.random.randint(0, 256))  # Random color
    thickness = np.random.randint(1, 10)
    cv2.line(image, pt1, pt2, color, thickness)

    # Save the image
    filename = os.path.join(output_folder, f"image_{i + 1}.png")
    cv2.imwrite(filename, image)

    print(f"Saved {filename}")

print("All images generated and saved in the 'output' folder.")
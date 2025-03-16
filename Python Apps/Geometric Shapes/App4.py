import cv2
import numpy as np
import os

# Create the output folder if it doesn't exist
output_folder = "output"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Image dimensions
width, height = 500, 500

# Function to rotate a shape
def rotate_shape(image, shape, angle, center):
    """
    Rotates a shape around its center.
    """
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated_image = cv2.warpAffine(image, rotation_matrix, (width, height), borderValue=(255, 255, 255))
    return rotated_image

# Generate 50 images
for i in range(50):
    # Create a blank image (white background)
    image = np.ones((height, width, 3), dtype=np.uint8) * 255  # White background

    # Draw random shapes
    for _ in range(np.random.randint(3, 6)):  # Draw 3 to 5 random shapes per image
        shape = np.random.choice(["circle", "square", "rectangle", "ellipse"])

        # Random color
        color = (np.random.randint(0, 256), np.random.randint(0, 256), np.random.randint(0, 256))

        if shape == "circle":
            # Draw a random circle (larger radius)
            center = (np.random.randint(0, width), np.random.randint(0, height))
            radius = np.random.randint(50, 150)  # Larger circles
            cv2.circle(image, center, radius, color, -1)  # Filled circle

        elif shape == "square":
            # Draw a random rotated square
            side = np.random.randint(50, 150)
            top_left = (np.random.randint(0, width - side), np.random.randint(0, height - side))
            bottom_right = (top_left[0] + side, top_left[1] + side)

            # Create a temporary image to draw the square
            temp_image = np.ones((height, width, 3), dtype=np.uint8) * 255
            cv2.rectangle(temp_image, top_left, bottom_right, color, -1)

            # Rotate the square
            center = ((top_left[0] + bottom_right[0]) // 2, (top_left[1] + bottom_right[1]) // 2)
            angle = np.random.randint(0, 360)  # Random rotation angle
            rotated_image = rotate_shape(temp_image, "square", angle, center)

            # Combine the rotated square with the main image
            image = cv2.bitwise_and(image, rotated_image)

        elif shape == "rectangle":
            # Draw a random rotated rectangle
            w = np.random.randint(50, 200)
            h = np.random.randint(50, 200)
            top_left = (np.random.randint(0, width - w), np.random.randint(0, height - h))
            bottom_right = (top_left[0] + w, top_left[1] + h)

            # Create a temporary image to draw the rectangle
            temp_image = np.ones((height, width, 3), dtype=np.uint8) * 255
            cv2.rectangle(temp_image, top_left, bottom_right, color, -1)

            # Rotate the rectangle
            center = ((top_left[0] + bottom_right[0]) // 2, (top_left[1] + bottom_right[1]) // 2)
            angle = np.random.randint(0, 360)  # Random rotation angle
            rotated_image = rotate_shape(temp_image, "rectangle", angle, center)

            # Combine the rotated rectangle with the main image
            image = cv2.bitwise_and(image, rotated_image)

        elif shape == "ellipse":
            # Draw a random rotated ellipse
            center = (np.random.randint(0, width), np.random.randint(0, height))
            axes = (np.random.randint(50, 150), np.random.randint(50, 150))
            angle = np.random.randint(0, 360)  # Random rotation angle
            cv2.ellipse(image, center, axes, angle, 0, 360, color, -1)  # Filled ellipse

    # Save the image
    filename = os.path.join(output_folder, f"image_{i + 1}.png")
    cv2.imwrite(filename, image)

    print(f"Saved {filename}")

print("All images generated and saved in the 'output' folder.")
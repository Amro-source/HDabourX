import cv2
import numpy as np
import os
import random

# Create the output folder if it doesn't exist
output_folder = "output"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Image dimensions
width, height = 500, 500

# Function to check if two rectangles overlap
def rectangles_overlap(rect1, rect2):
    """
    Check if two rectangles overlap.
    Each rectangle is represented as (x1, y1, x2, y2).
    """
    return not (rect1[2] < rect2[0] or rect1[0] > rect2[2] or rect1[3] < rect2[1] or rect1[1] > rect2[3])

# Function to get the bounding box of a shape
def get_bounding_box(shape, params):
    """
    Get the bounding box (x1, y1, x2, y2) of a shape.
    """
    if shape == "circle":
        center, radius = params
        x1, y1 = center[0] - radius, center[1] - radius
        x2, y2 = center[0] + radius, center[1] + radius
        return (x1, y1, x2, y2)
    elif shape in ["square", "rectangle"]:
        top_left, bottom_right = params
        return (top_left[0], top_left[1], bottom_right[0], bottom_right[1])
    elif shape == "ellipse":
        center, axes, _ = params
        x1, y1 = center[0] - axes[0], center[1] - axes[1]
        x2, y2 = center[0] + axes[0], center[1] + axes[1]
        return (x1, y1, x2, y2)

# Generate 50 images
for i in range(50):
    # Create a blank image (white background)
    image = np.ones((height, width, 3), dtype=np.uint8) * 255  # White background

    # List to track occupied areas
    occupied_areas = []

    # Draw random shapes
    for _ in range(np.random.randint(3, 6)):  # Draw 3 to 5 random shapes per image
        while True:
            shape = random.choice(["circle", "square", "rectangle", "ellipse"])
            color = (random.randint(0, 256), random.randint(0, 256), random.randint(0, 256))  # Random color

            if shape == "circle":
                # Random circle parameters
                radius = random.randint(50, 100)  # Larger circles
                center = (random.randint(radius, width - radius), random.randint(radius, height - radius))
                bounding_box = get_bounding_box("circle", (center, radius))

            elif shape == "square":
                # Random square parameters
                side = random.randint(50, 150)
                top_left = (random.randint(0, width - side), random.randint(0, height - side))
                bottom_right = (top_left[0] + side, top_left[1] + side)
                bounding_box = get_bounding_box("square", (top_left, bottom_right))

            elif shape == "rectangle":
                # Random rectangle parameters
                w = random.randint(50, 200)
                h = random.randint(50, 200)
                top_left = (random.randint(0, width - w), random.randint(0, height - h))
                bottom_right = (top_left[0] + w, top_left[1] + h)
                bounding_box = get_bounding_box("rectangle", (top_left, bottom_right))

            elif shape == "ellipse":
                # Random ellipse parameters
                axes = (random.randint(50, 150), random.randint(50, 150))
                center = (random.randint(axes[0], width - axes[0]), random.randint(axes[1], height - axes[1]))
                angle = random.randint(0, 360)  # Random rotation angle
                bounding_box = get_bounding_box("ellipse", (center, axes, angle))

            # Check if the new shape overlaps with any existing shape
            overlap = False
            for occupied in occupied_areas:
                if rectangles_overlap(bounding_box, occupied):
                    overlap = True
                    break

            if not overlap:
                break  # Exit the loop if no overlap is found

        # Draw the shape
        if shape == "circle":
            cv2.circle(image, center, radius, color, -1)  # Filled circle
        elif shape == "square":
            cv2.rectangle(image, top_left, bottom_right, color, -1)  # Filled square
        elif shape == "rectangle":
            cv2.rectangle(image, top_left, bottom_right, color, -1)  # Filled rectangle
        elif shape == "ellipse":
            cv2.ellipse(image, center, axes, angle, 0, 360, color, -1)  # Filled ellipse

        # Add the bounding box to the occupied areas list
        occupied_areas.append(bounding_box)

    # Save the image
    filename = os.path.join(output_folder, f"image_{i + 1}.png")
    cv2.imwrite(filename, image)

    print(f"Saved {filename}")

print("All images generated and saved in the 'output' folder.")
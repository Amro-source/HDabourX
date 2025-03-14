import cv2
import numpy as np

# Create a blank image (white background)
width, height = 500, 500
image = np.ones((height, width, 3), dtype=np.uint8) * 255  # White background

# Draw a blue rectangle
cv2.rectangle(image, (50, 50), (200, 200), (255, 0, 0), -1)  # -1 fills the rectangle

# Draw a green circle
cv2.circle(image, (300, 300), 50, (0, 255, 0), -1)  # -1 fills the circle

# Draw a red line
cv2.line(image, (400, 100), (450, 400), (0, 0, 255), 5)

# Display the image
cv2.imshow("Geometric Shapes", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Save the image
cv2.imwrite("geometric_shapes.png", image)
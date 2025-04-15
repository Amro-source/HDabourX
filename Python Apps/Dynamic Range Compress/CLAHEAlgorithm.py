import cv2
import numpy as np

def clahe(image):
    """
    Applies CLAHE to an image.

    Args:
        image (numpy array): The input image.

    Returns:
        numpy array: The image with CLAHE applied.
    """
    # Convert to LAB color space
    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

    # Apply CLAHE to the L channel
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    lab_image[:, :, 0] = clahe.apply(lab_image[:, :, 0])

    # Convert back to BGR color space
    clahe_image = cv2.cvtColor(lab_image, cv2.COLOR_LAB2BGR)

    return clahe_image


def display_image(window_name, image):
    """
    Displays an image in a window, resizing it if necessary.

    Args:
        window_name (str): The name of the window.
        image (numpy array): The image to display.
    """
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    max_size = 800  # Maximum size of the window
    h, w = image.shape[:2]
    scale = min(max_size / w, max_size / h)
    if scale < 1:
        new_size = (int(w * scale), int(h * scale))
        cv2.resizeWindow(window_name, new_size)
    cv2.imshow(window_name, image)


def main():
    # Load the image
    image_path = "Normal (2).jpg"
    image_path = "Whale.jpg"
    image = cv2.imread(image_path)

    # Apply CLAHE
    clahe_image = clahe(image)

  # Display the original and CLAHE images
    display_image("Original Image", image)
    display_image("CLAHE Image", clahe_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()


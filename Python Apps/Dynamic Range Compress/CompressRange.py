import cv2
import numpy as np

def compress_dynamic_range(image, alpha, beta):
    """
    Compresses the dynamic range of an image.

    Args:
        image (numpy array): The input image.
        alpha (float): Contrast control (1.0-3.0).
        beta (int): Brightness control (0-100).

    Returns:
        numpy array: The image with compressed dynamic range.
    """
    # Apply contrast and brightness adjustments
    adjusted_image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

    return adjusted_image

def main():
    # Load the image
    image_path = "path_to_your_image.jpg"
    image = cv2.imread(image_path)

    # Compress dynamic range
    alpha = 0.5  # Contrast reduction
    beta = 0     # No brightness adjustment
    compressed_image = compress_dynamic_range(image, alpha, beta)

    # Display the original and compressed images
    cv2.imshow("Original Image", image)
    cv2.imshow("Compressed Dynamic Range Image", compressed_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Save the compressed image
    cv2.imwrite("compressed_image.jpg", compressed_image)

if __name__ == "__main__":
    main()

import cv2
import numpy as np

def histogram_equalization(image):
    """
    Applies histogram equalization to an image.

    Args:
        image (numpy array): The input image.

    Returns:
        numpy array: The image with histogram equalization applied.
    """
    # Convert to YCrCb color space
    ycrcb_image = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)

    # Apply histogram equalization to the Y channel
    ycrcb_image[:, :, 0] = cv2.equalizeHist(ycrcb_image[:, :, 0])

    # Convert back to BGR color space
    equalized_image = cv2.cvtColor(ycrcb_image, cv2.COLOR_YCrCb2BGR)

    return equalized_image

def main():
    # Load the image
    image_path = "path_to_your_image.jpg"
    image = cv2.imread(image_path)

    # Apply histogram equalization
    equalized_image = histogram_equalization(image)

    # Display the original and equalized images
    cv2.imshow("Original Image", image)
    cv2.imshow("Histogram Equalized Image", equalized_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

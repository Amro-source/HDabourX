#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

int main() {
    // Load an image from file
    // Replace "images/tiger.jpg" with your actual image path
    Mat image = imread("tiger.jpg", IMREAD_COLOR);

    // Check if image loaded successfully
    if (image.empty()) {
        cout << "Could not open or find the image" << endl;
        return -1;
    }

    // Create a window for display
    namedWindow("Tiger Image", WINDOW_AUTOSIZE);

    // Show our image inside the window
    imshow("Tiger Image", image);

    // Wait for a keystroke in the window
    waitKey(0);

    return 0;
}
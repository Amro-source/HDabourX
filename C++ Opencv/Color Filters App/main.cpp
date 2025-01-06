#include <opencv2/opencv.hpp>
#include <iostream>

#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/utils/logger.hpp>
#include <iostream>

// Function to apply a red filter
cv::Mat applyRedFilter(const cv::Mat& img) {
    cv::Mat redFiltered = cv::Mat::zeros(img.size(), img.type());
    std::vector<cv::Mat> channels;
    cv::split(img, channels);
    channels[1] = cv::Mat::zeros(img.rows, img.cols, channels[1].type()); // Set green channel to 0
    channels[2] = cv::Mat::zeros(img.rows, img.cols, channels[2].type()); // Set blue channel to 0
    cv::merge(channels, redFiltered);
    return redFiltered;
}

// Function to apply a green filter
cv::Mat applyGreenFilter(const cv::Mat& img) {
    cv::Mat greenFiltered = cv::Mat::zeros(img.size(), img.type());
    std::vector<cv::Mat> channels;
    cv::split(img, channels);
    channels[0] = cv::Mat::zeros(img.rows, img.cols, channels[0].type()); // Set red channel to 0
    channels[2] = cv::Mat::zeros(img.rows, img.cols, channels[2].type()); // Set blue channel to 0
    cv::merge(channels, greenFiltered);
    return greenFiltered;
}

// Function to apply a blue filter
cv::Mat applyBlueFilter(const cv::Mat& img) {
    cv::Mat blueFiltered = cv::Mat::zeros(img.size(), img.type());
    std::vector<cv::Mat> channels;
    cv::split(img, channels);
    channels[0] = cv::Mat::zeros(img.rows, img.cols, channels[0].type()); // Set red channel to 0
    channels[1] = cv::Mat::zeros(img.rows, img.cols, channels[1].type()); // Set green channel to 0
    cv::merge(channels, blueFiltered);
    return blueFiltered;
}

int main() {
    // Set OpenCV log level to silent
    cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_SILENT);
    
    
    
    // Load the image
    std::string imagePath = "D:/Gallery/tiger.jpg";
    cv::Mat img = cv::imread(imagePath);

    if (img.empty()) {
        std::cerr << "Error: Could not load image at " << imagePath << std::endl;
        return -1;
    }

    // Apply filters
    cv::Mat redFiltered = applyRedFilter(img);
    cv::Mat greenFiltered = applyGreenFilter(img);
    cv::Mat blueFiltered = applyBlueFilter(img);

    // Display the original and filtered images
    cv::imshow("Original Image", img);
    cv::imshow("Red Filter", redFiltered);
    cv::imshow("Green Filter", greenFiltered);
    cv::imshow("Blue Filter", blueFiltered);

    // Wait for a key press and close all windows
    cv::waitKey(0);
    cv::destroyAllWindows();

    return 0;
}

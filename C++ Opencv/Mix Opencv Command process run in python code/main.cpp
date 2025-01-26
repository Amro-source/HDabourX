#include <iostream>
#include <filesystem>
#include <opencv2/opencv.hpp>

namespace fs = std::filesystem;
using namespace cv;

Mat apply_blue_filter(const Mat& image) {
    Mat filtered_image;
    std::vector<Mat> channels(3);
    split(image, channels);

    // Set green and red channels to zero
    channels[1] = Mat::zeros(image.rows, image.cols, CV_8UC1);
    channels[2] = Mat::zeros(image.rows, image.cols, CV_8UC1);

    merge(channels, filtered_image);
    return filtered_image;
}

int main(int argc, char* argv[]) {
    if (argc != 3) {
        std::cerr << "Usage: <executable> <input_folder> <output_folder>" << std::endl;
        return -1;
    }

    std::string input_folder = argv[1];
    std::string output_folder = argv[2];

    // Ensure the output folder exists
    fs::create_directories(output_folder);

    // Process all images in the input folder
    for (const auto& entry : fs::directory_iterator(input_folder)) {
        if (entry.is_regular_file()) {
            std::string input_path = entry.path().string();
            std::string output_path = output_folder + "/processed_" + entry.path().filename().string();

            std::cout << "Processing: " << input_path << std::endl;

            Mat image = imread(input_path, IMREAD_COLOR);
            if (image.empty()) {
                std::cerr << "Error: Could not read image: " << input_path << std::endl;
                continue;
            }

            Mat blue_image = apply_blue_filter(image);

            if (!imwrite(output_path, blue_image)) {
                std::cerr << "Error: Could not save image to: " << output_path << std::endl;
                continue;
            }

            std::cout << "Saved: " << output_path << std::endl;
        }
    }

    std::cout << "All images processed successfully!" << std::endl;
    return 0;
}

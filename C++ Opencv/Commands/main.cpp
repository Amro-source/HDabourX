// C++ application using OpenCV
#include <opencv2/opencv.hpp>
#include <iostream>
#include <filesystem>
#include <string>
#include <fstream>
#include <sstream>
//#include <experimental/filesystem>
#include <filesystem>

//namespace fs = std::experimental::filesystem;
#define _SILENCE_EXPERIMENTAL_FILESYSTEM_DEPRECATION_WARNING
namespace fs = std::filesystem;

void applyGrayscaleFilter(const std::string& inputPath, const std::string& outputPath) {
    cv::Mat image = cv::imread(inputPath);
    if (image.empty()) {
        std::cerr << "Failed to load image: " << inputPath << std::endl;
        return;
    }

    cv::Mat grayscale;
    cv::cvtColor(image, grayscale, cv::COLOR_BGR2GRAY);
    cv::imwrite(outputPath, grayscale);
}

void processImages(const std::string& inputDir, const std::string& outputDir) {
    for (const auto& entry : fs::directory_iterator(inputDir)) {
        if (fs::status(entry).type() == fs::file_type::regular) {
            std::string inputPath = entry.path().string();
            std::string outputPath = outputDir + "/" + entry.path().filename().string();
            applyGrayscaleFilter(inputPath, outputPath);
        }
    }
}

int main() {
    const std::string STAGE_1 = "STAGE_1";
    const std::string STAGE_3 = "STAGE_3";
    const std::string COMMAND_FILE = "command.txt";
    const std::string RESPONSE_FILE = "response.txt";

    std::cout << "Server is running, waiting for Python client...\n";

    while (true) {
        std::ifstream commandStream(COMMAND_FILE);
        if (!commandStream.is_open()) {
            std::cerr << "Failed to open command file.\n";
            continue;
        }

        std::string imageName;
        std::getline(commandStream, imageName);
        commandStream.close();

        if (imageName.empty()) {
            continue;
        }

        std::string inputPath = STAGE_1 + "/" + imageName;
        std::string outputPath = STAGE_3 + "/" + imageName;

        applyGrayscaleFilter(inputPath, outputPath);

        std::ofstream responseStream(RESPONSE_FILE);
        if (!responseStream.is_open()) {
            std::cerr << "Failed to open response file.\n";
            continue;
        }

        responseStream << "Processed: " << imageName;
        responseStream.close();

        // Clear the command file to signal completion
        std::ofstream clearCommand(COMMAND_FILE, std::ios::trunc);
        clearCommand.close();
    }

    return 0;
}
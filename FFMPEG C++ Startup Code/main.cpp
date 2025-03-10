extern "C" {
#include <libavformat/avformat.h>
#include <libavutil/error.h>
}

#include <iostream>

int main(int argc, char* argv[]) {
    // Initialize FFmpeg (network support)
    avformat_network_init();

    // Open the input file
    AVFormatContext* formatContext = nullptr;
    const char* inputFilename = "input.mp4";  // Replace with your file

    // Error handling for file opening
    if (avformat_open_input(&formatContext, inputFilename, nullptr, nullptr) < 0) {
        std::cerr << "Could not open file: " << inputFilename << std::endl;
        return -1;
    }

    // Retrieve stream information
    if (avformat_find_stream_info(formatContext, nullptr) < 0) {
        std::cerr << "Could not find stream information" << std::endl;
        avformat_close_input(&formatContext);
        return -1;
    }

    // Print file information
    std::cout << "File: " << inputFilename << std::endl;
    std::cout << "Format: " << formatContext->iformat->name << std::endl;
    std::cout << "Duration: " << formatContext->duration << " ("
        << static_cast<double>(formatContext->duration) / AV_TIME_BASE << " seconds)" << std::endl;
    std::cout << "Streams: " << formatContext->nb_streams << std::endl;

    // Clean up
    avformat_close_input(&formatContext);
    return 0;
}
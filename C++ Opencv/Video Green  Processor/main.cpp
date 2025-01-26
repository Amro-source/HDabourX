#include <opencv2/opencv.hpp>
#include <iostream>

using namespace std;
using namespace cv;

void applyGreenFilter(Mat& frame) {
    // Set the red and blue channels to zero
    frame.forEach<Vec3b>([](Vec3b& pixel, const int* position) -> void {
        pixel[0] = 0; // Blue channel
        pixel[2] = 0; // Red channel
        });
}

void processVideo(const string& inputFile, const string& outputFile) {
    // Open the video file
    VideoCapture cap(inputFile);
    if (!cap.isOpened()) {
        cout << "Error: Unable to open video file " << inputFile << endl;
        return;
    }

    // Get video properties
    int width = static_cast<int>(cap.get(CAP_PROP_FRAME_WIDTH));
    int height = static_cast<int>(cap.get(CAP_PROP_FRAME_HEIGHT));
    double fps = cap.get(CAP_PROP_FPS);
    int fourcc = VideoWriter::fourcc('X', 'V', 'I', 'D');

    // Define the codec and create a VideoWriter object
    VideoWriter out(outputFile, fourcc, fps, Size(width, height));

    Mat frame;
    int frameNumber = 0;
    while (true) {
        // Read a frame
        cap >> frame;
        if (frame.empty()) {
            break; // Exit the loop if no more frames
        }

        // Increment frame counter
        frameNumber++;
        cout << "Processing frame #" << frameNumber << endl;

        // Apply the green filter
        applyGreenFilter(frame);

        // Write the frame to the output file
        out.write(frame);
    }

    // Release resources
    cap.release();
    out.release();
    cout << "Video processing complete. Output saved to " << outputFile << endl;
}

int main() {
    string inputVideo = "D:/Python/Video Processor/input_video.mp4";  // Change to your input file path
    string outputVideo = "output_video.avi"; // Change to your desired output file path

    processVideo(inputVideo, outputVideo);
    return 0;
}
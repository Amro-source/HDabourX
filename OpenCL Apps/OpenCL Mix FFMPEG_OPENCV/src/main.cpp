#include <iostream>
#include <fstream>
#include <vector>
#include <CL/cl.h>  // OpenCL C API
#include <opencv2/opencv.hpp>  // OpenCV
extern "C" {
#include <libavformat/avformat.h>  // FFmpeg
}

// Helper function to check OpenCL errors
void checkError(cl_int err, const char* operation) {
    if (err != CL_SUCCESS) {
        std::cerr << "Error during " << operation << ": " << err << std::endl;
        exit(EXIT_FAILURE);
    }
}

int main() {
    // FFmpeg: No need for av_register_all() in modern FFmpeg versions
    std::cout << "FFmpeg initialized (no explicit registration required)." << std::endl;

    // OpenCV example: Load an image
    cv::Mat image = cv::imread("example.jpg");
    if (image.empty()) {
        std::cerr << "Failed to load image!" << std::endl;
        return -1;
    }
    std::cout << "OpenCV loaded image: " << image.size() << std::endl;

    // OpenCL setup
    cl_int err;

    // Get platforms
    cl_uint platformCount;
    err = clGetPlatformIDs(0, nullptr, &platformCount);
    checkError(err, "clGetPlatformIDs");

    std::vector<cl_platform_id> platforms(platformCount);
    err = clGetPlatformIDs(platformCount, platforms.data(), nullptr);
    checkError(err, "clGetPlatformIDs");

    if (platformCount == 0) {
        std::cerr << "No OpenCL platforms found!" << std::endl;
        return -1;
    }

    cl_platform_id platform = platforms[0];
    size_t platformNameSize;
    err = clGetPlatformInfo(platform, CL_PLATFORM_NAME, 0, nullptr, &platformNameSize);
    checkError(err, "clGetPlatformInfo");

    std::vector<char> platformName(platformNameSize);
    err = clGetPlatformInfo(platform, CL_PLATFORM_NAME, platformNameSize, platformName.data(), nullptr);
    checkError(err, "clGetPlatformInfo");

    std::cout << "Using OpenCL platform: " << platformName.data() << std::endl;

    // Get devices
    cl_uint deviceCount;
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 0, nullptr, &deviceCount);
    checkError(err, "clGetDeviceIDs");

    if (deviceCount == 0) {
        std::cerr << "No GPU devices found!" << std::endl;
        return -1;
    }

    std::vector<cl_device_id> devices(deviceCount);
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, deviceCount, devices.data(), nullptr);
    checkError(err, "clGetDeviceIDs");

    cl_device_id device = devices[0];
    size_t deviceNameSize;
    err = clGetDeviceInfo(device, CL_DEVICE_NAME, 0, nullptr, &deviceNameSize);
    checkError(err, "clGetDeviceInfo");

    std::vector<char> deviceName(deviceNameSize);
    err = clGetDeviceInfo(device, CL_DEVICE_NAME, deviceNameSize, deviceName.data(), nullptr);
    checkError(err, "clGetDeviceInfo");

    std::cout << "Using OpenCL device: " << deviceName.data() << std::endl;

    // Create context and command queue
    cl_context context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &err);
    checkError(err, "clCreateContext");

    cl_command_queue queue = clCreateCommandQueue(context, device, 0, &err);
    checkError(err, "clCreateCommandQueue");

    // Load OpenCL kernel from file
    std::ifstream kernelFile("kernels.cl");
    if (!kernelFile.is_open()) {
        std::cerr << "Failed to open kernel file!" << std::endl;
        return -1;
    }

    std::string kernelSource((std::istreambuf_iterator<char>(kernelFile)), std::istreambuf_iterator<char>());
    const char* sourcePtr = kernelSource.c_str();
    size_t sourceSize = kernelSource.size();

    // Create and build program
    cl_program program = clCreateProgramWithSource(context, 1, &sourcePtr, &sourceSize, &err);
    checkError(err, "clCreateProgramWithSource");

    err = clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        size_t logSize;
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &logSize);

        std::vector<char> buildLog(logSize);
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, logSize, buildLog.data(), nullptr);

        std::cerr << "Build log:\n" << buildLog.data() << std::endl;
        return -1;
    }

    // Create kernel object
    cl_kernel kernel = clCreateKernel(program, "example_kernel", &err);
    checkError(err, "clCreateKernel");

    // TODO: Add buffer creation, kernel arguments, and execution here

    // Cleanup
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

    return 0;
}
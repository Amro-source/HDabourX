cmake_minimum_required(VERSION 3.14)
project(MyFFmpegOpenCVProject LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# FFmpeg configuration
set(FFMPEG_ROOT "C:/ffmpeg-master-latest-win64-gpl-shared")  # Use forward slashes
include_directories(${FFMPEG_ROOT}/include)

# OpenCV configuration
find_package(OpenCV REQUIRED)  # Automatically finds OpenCV installation
if(NOT OpenCV_FOUND)
    message(FATAL_ERROR "OpenCV not found! Install OpenCV or set OpenCV_DIR.")
endif()

# OpenCL configuration
find_package(OpenCL REQUIRED)  # Find OpenCL package
if(NOT OpenCL_FOUND)
    message(FATAL_ERROR "OpenCL not found! Install OpenCL SDK or ensure it's in your PATH.")
endif()

# Add your executable
add_executable(${PROJECT_NAME} main.cpp)

# Link FFmpeg libraries (use PRIVATE keyword)
target_link_libraries(${PROJECT_NAME}
    PRIVATE
        ${FFMPEG_ROOT}/lib/avformat.lib
        ${FFMPEG_ROOT}/lib/avcodec.lib
        ${FFMPEG_ROOT}/lib/avutil.lib
        ${FFMPEG_ROOT}/lib/swscale.lib
        ws2_32   # Windows networking
        bcrypt
        secur32
)

# Link OpenCV libraries (use PRIVATE keyword)
target_link_libraries(${PROJECT_NAME}
    PRIVATE
        ${OpenCV_LIBS}  # Links all OpenCV modules
)

# Link OpenCL library (use PRIVATE keyword)
target_link_libraries(${PROJECT_NAME}
    PRIVATE
        OpenCL::OpenCL  # Links OpenCL library
)

# Include directories
target_include_directories(${PROJECT_NAME}
    PRIVATE
        ${FFMPEG_ROOT}/include
        ${OpenCV_INCLUDE_DIRS}
        ${OpenCL_INCLUDE_DIRS}  # Include OpenCL headers
)

# Copy the OpenCL kernel file to the build directory (if needed)
configure_file(src/kernels.cl ${CMAKE_BINARY_DIR}/kernels.cl COPYONLY)
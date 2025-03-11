cmake_minimum_required(VERSION 3.14)
project(MyFFmpegOpenCVProject LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)

# FFmpeg configuration
set(FFMPEG_ROOT "C:/ffmpeg-master-latest-win64-gpl-shared")  # Use forward slashes
include_directories(${FFMPEG_ROOT}/include)

# OpenCV configuration
find_package(OpenCV REQUIRED)  # Automatically finds OpenCV installation
if(NOT OpenCV_FOUND)
    message(FATAL_ERROR "OpenCV not found! Install OpenCV or set OpenCV_DIR.")
endif()

# Add your executable
add_executable(${PROJECT_NAME} main.cpp)

# Link FFmpeg libraries
target_link_libraries(${PROJECT_NAME}
    ${FFMPEG_ROOT}/lib/avformat.lib
    ${FFMPEG_ROOT}/lib/avcodec.lib
    ${FFMPEG_ROOT}/lib/avutil.lib
    ${FFMPEG_ROOT}/lib/swscale.lib
    # Add other FFmpeg libraries as needed
    ws2_32   # Windows networking
    bcrypt
    secur32
)

# Link OpenCV libraries
target_link_libraries(${PROJECT_NAME}
    ${OpenCV_LIBS}  # Links all OpenCV modules
)

# Include OpenCV headers
include_directories(${OpenCV_INCLUDE_DIRS})
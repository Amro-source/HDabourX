cmake_minimum_required(VERSION 3.14)
project(MyFFmpegProject LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)

# Fixed path with double backslashes
set(FFMPEG_ROOT "C:\\ffmpeg-master-latest-win64-gpl-shared")

# Include FFmpeg headers
include_directories(${FFMPEG_ROOT}/include)

# Link FFmpeg libraries
add_executable(${PROJECT_NAME} main.cpp)
target_link_libraries(${PROJECT_NAME}
    ${FFMPEG_ROOT}/lib/avformat.lib
    ${FFMPEG_ROOT}/lib/avcodec.lib
    ${FFMPEG_ROOT}/lib/avutil.lib
    ${FFMPEG_ROOT}/lib/swscale.lib
    # Add other FFmpeg libraries as needed
    ws2_32  # Required for networking on Windows
    bcrypt
    secur32
)
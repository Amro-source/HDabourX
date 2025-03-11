# Specify the minimum required version of CMake
cmake_minimum_required(VERSION 3.10)

# Define the project name and language
project(OpenCLTutorial LANGUAGES CXX)

# Set the C++ standard to C++17 (or higher if needed)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Find the OpenCL package
find_package(OpenCL REQUIRED)

# Add the executable target
add_executable(OpenCLTutorial src/main.cpp)

# Link the OpenCL library to the target
target_link_libraries(OpenCLTutorial PRIVATE OpenCL::OpenCL)

# Include the OpenCL headers
target_include_directories(OpenCLTutorial PRIVATE ${OpenCL_INCLUDE_DIRS})

# Copy the OpenCL kernel file to the build directory
configure_file(src/kernels.cl ${CMAKE_BINARY_DIR}/kernels.cl COPYONLY)
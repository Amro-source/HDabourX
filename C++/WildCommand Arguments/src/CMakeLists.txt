cmake_minimum_required(VERSION 3.10)

# Set the project name
project(CommandLineTool)

# Specify the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Find the dirent library
find_path(DIRENT_INCLUDE_DIRS "dirent.h")

# Add the executable
add_executable(command_line_tool main.cpp)

# Include the dirent headers
if(DIRENT_INCLUDE_DIRS)
  target_include_directories(command_line_tool PRIVATE ${DIRENT_INCLUDE_DIRS})
else()
  message(FATAL_ERROR "dirent.h not found")
endif()

README: Video Green Filter Application 
Overview 

This application processes a video file by applying a green filter to each frame. The filter retains only the green channel of the RGB color space, setting the red and blue channels to zero. The application supports two processing modes: 

    CPU-based processing  using OpenCV.
    GPU-based processing  using OpenCL.
     

The processed video is saved as a new file, with separate outputs for CPU and GPU processing. The application also measures and reports the processing time for both modes. 
Features 

    Green Filter : Retains only the green channel of the RGB image.
    Dual Processing Modes :
        CPU processing using OpenCV for high-level image manipulation.
        GPU processing using OpenCL for parallel computation.
         
    Video Encoding : Encodes the processed frames into an output video file using FFmpeg.
    Performance Measurement : Measures and compares the execution time of CPU and GPU processing.
     

Dependencies 

To compile and run this application, you need the following libraries and tools: 
Required Libraries 

    FFmpeg :
        Used for video decoding, encoding, and frame manipulation.
        Install via package manager (e.g., apt install ffmpeg on Ubuntu) or build from source.
         
    OpenCV :
        Used for CPU-based image processing.
        Install via package manager (e.g., apt install libopencv-dev) or build from source.
         
    OpenCL :
        Used for GPU-based image processing.
        Requires an OpenCL-compatible GPU and drivers.
        Install the OpenCL SDK for your platform (e.g., NVIDIA CUDA Toolkit, AMD ROCm, or Intel OpenCL Runtime).
         
     

Build Tools 

    A C++ compiler (e.g., GCC or Clang).
    CMake (optional, for building the project).
     

Compilation Instructions 

    Install Dependencies :
    Ensure all required libraries are installed on your system. For example: 

sudo apt update
sudo apt install ffmpeg libopencv-dev ocl-icd-opencl-dev build-essential

Compile the Application :
Use the following commands to compile the application:
g++ -o green_filter_app main.cpp \
    `pkg-config --cflags --libs opencv4` \
    -lavformat -lavcodec -lavutil -lswscale -lstdc++fs -lOpenCL

Replace main.cpp with the actual source file name if different. 

Verify Installation :
Run the compiled binary to ensure it works: 

./green_filter_app

Usage 
Input File 

Place the input video file (input.mp4) in the same directory as the application or specify its path in the INPUT_FILE constant in the source code. 
Running the Application 

Execute the applicati

on without arguments: 
./green_filter_app

Output Files 

    CPU-Processed Video : Saved as output_cpu.mp4.
    GPU-Processed Video : Saved as output_gpu.mp4.
     

Performance Report 

After processing, the application prints the execution times for both CPU and GPU modes: 
CPU Processing Time: X.XXX seconds
GPU Processing Time: Y.YYY seconds

Code Structure 
Key Functions 

    apply_green_filter_cpu :
    Applies the green filter using OpenCV on the CPU. 

    init_opencl :
    Initializes the OpenCL context, device, and kernel for GPU processing. 

    apply_green_filter_gpu :
    Applies the green filter using OpenCL on the GPU. 

    process_frames :
    Decodes video frames, applies the filter (CPU or GPU), and encodes the processed frames into an output file. 

    init_decoder and init_encoder :
    Initialize FFmpeg's decoder and encoder for video processing. 

    main :
    Orchestrates the entire workflow, including initialization, processing, and cleanup. 
     

Optimization Notes 
GPU Performance 

If GPU processing is slower than CPU processing, consider the following optimizations: 

    Minimize Data Transfers :
    Reduce the frequency of data transfers between the CPU and GPU.
    Optimize OpenCL Kernel :
    Use shared memory and vectorized operations to improve GPU efficiency.
    Profile the Application :
    Use profiling tools (e.g., NVIDIA Nsight, Intel VTune) to identify bottlenecks.
     

CPU Performance 

The CPU implementation uses OpenCV, which is highly optimized. However, you can experiment with SIMD instructions or multi-threading for further improvements. 
Known Issues 

    Initialization Overhead :
    OpenCL initialization and kernel compilation may dominate runtime for short videos.
    Hardware Dependency :
    GPU performance depends on the hardware specifications. Ensure you have a capable GPU and up-to-date drivers.
    Error Handling :
    The application assumes valid input files and hardware configurations. Add robust error handling for production use.
     
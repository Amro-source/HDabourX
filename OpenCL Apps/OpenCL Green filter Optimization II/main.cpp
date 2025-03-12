#include <opencv2/opencv.hpp>
#include <CL/cl.h> // OpenCL C API
#include <chrono>
#include <iostream>

extern "C" {
#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
#include <libavutil/avutil.h>
#include <libavutil/imgutils.h>
#include <libavutil/opt.h>
#include <libswscale/swscale.h>
}

const char* INPUT_FILE = "input.mp4";
const char* OUTPUT_CPU_FILE = "output_cpu.mp4";
const char* OUTPUT_GPU_FILE = "output_gpu.mp4";

// Function prototypes
bool init_decoder(AVFormatContext*& fmt_ctx, AVCodecContext*& dec_ctx, int& video_stream_idx);
bool init_encoder(AVFormatContext*& fmt_ctx, AVCodecContext*& enc_ctx, AVStream*& out_stream, AVRational input_time_base, int width, int height, const char* output_file);

// Apply green filter using OpenCV (CPU)
void apply_green_filter_cpu(cv::Mat& frame) {
    std::vector<cv::Mat> channels;
    cv::split(frame, channels);
    channels[0] = cv::Scalar(0);  // Zero out Blue
    channels[2] = cv::Scalar(0);  // Zero out Red
    cv::merge(channels, frame);
}

// Initialize OpenCL context and kernel
cl_context init_opencl(cl_device_id& device, cl_command_queue& queue, cl_kernel& kernel) {
    cl_int err;

    // Get platform
    cl_platform_id platform;
    err = clGetPlatformIDs(1, &platform, nullptr);
    if (err != CL_SUCCESS) {
        std::cerr << "Failed to get OpenCL platform: " << err << std::endl;
        return nullptr;
    }
    std::cout << "OpenCL platform found!" << std::endl;

    // Get device
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, nullptr);
    if (err != CL_SUCCESS) {
        std::cerr << "Failed to get OpenCL device: " << err << std::endl;
        return nullptr;
    }
    std::cout << "OpenCL device found!" << std::endl;

    // Create context
    cl_context context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &err);
    if (err != CL_SUCCESS) {
        std::cerr << "Failed to create OpenCL context: " << err << std::endl;
        return nullptr;
    }
    std::cout << "OpenCL context created!" << std::endl;

    // Create command queue
    queue = clCreateCommandQueue(context, device, 0, &err);
    if (err != CL_SUCCESS) {
        std::cerr << "Failed to create OpenCL command queue: " << err << std::endl;
        return nullptr;
    }
    std::cout << "OpenCL command queue created!" << std::endl;

    // Load kernel source
    const char* kernelSource = R"(
    __kernel void green_filter(__global uchar* image, int width, int height) {
        int x = get_global_id(0);
        int y = get_global_id(1);
        if (x >= width || y >= height) return;
        
        int idx = (y * width + x) * 3; // 3 channels (BGR)
        image[idx] = 0;     // Zero out Blue
        image[idx + 2] = 0; // Zero out Red
    }
)";

    // Create program
    cl_program program = clCreateProgramWithSource(context, 1, &kernelSource, nullptr, &err);
    if (err != CL_SUCCESS) {
        std::cerr << "Failed to create OpenCL program: " << err << std::endl;
        return nullptr;
    }
    std::cout << "OpenCL program created!" << std::endl;

    // Build program
    err = clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        size_t logSize;
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &logSize);
        std::vector<char> buildLog(logSize);
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, logSize, buildLog.data(), nullptr);
        std::cerr << "OpenCL build log:\n" << buildLog.data() << std::endl;
        return nullptr;
    }
    std::cout << "OpenCL program built successfully!" << std::endl;

    // Create kernel
    kernel = clCreateKernel(program, "green_filter", &err);
    if (err != CL_SUCCESS) {
        std::cerr << "Failed to create OpenCL kernel: " << err << std::endl;
        return nullptr;
    }
    std::cout << "OpenCL kernel created!" << std::endl;

    return context;
}

// Apply green filter using OpenCL (GPU)
void apply_green_filter_gpu(cv::Mat& frame, cl_context context, cl_command_queue queue, cl_kernel kernel) {
    cl_int err;

    // Ensure the frame is continuous in memory
    cv::Mat contiguousFrame = frame.isContinuous() ? frame : frame.clone();
    //std::cout << "Frame is continuous: " << contiguousFrame.isContinuous() << std::endl;

    // Use pinned memory for faster transfer
    cl_mem buffer = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
        contiguousFrame.total() * contiguousFrame.elemSize(),
        nullptr, &err);
    if (err != CL_SUCCESS) {
        std::cerr << "Failed to create OpenCL buffer: " << err << std::endl;
        return;
    }
   // std::cout << "OpenCL buffer created successfully!" << std::endl;

    // Map memory for efficient transfer
    uchar* mappedPtr = (uchar*)clEnqueueMapBuffer(queue, buffer, CL_TRUE, CL_MAP_WRITE,
        0, contiguousFrame.total() * contiguousFrame.elemSize(),
        0, nullptr, nullptr, &err);
    if (err != CL_SUCCESS) {
        std::cerr << "Failed to map OpenCL buffer: " << err << std::endl;
        return;
    }
    //std::cout << "OpenCL buffer mapped successfully!" << std::endl;

    // Copy frame data to mapped OpenCL buffer
    memcpy(mappedPtr, contiguousFrame.data, contiguousFrame.total() * contiguousFrame.elemSize());
   // std::cout << "Frame data copied to OpenCL buffer!" << std::endl;

    // Unmap memory to sync data with GPU
    clEnqueueUnmapMemObject(queue, buffer, mappedPtr, 0, nullptr, nullptr);
    //std::cout << "OpenCL buffer unmapped successfully!" << std::endl;

    // Set kernel arguments
    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &buffer);
    err |= clSetKernelArg(kernel, 1, sizeof(int), &frame.cols);
    err |= clSetKernelArg(kernel, 2, sizeof(int), &frame.rows);
    if (err != CL_SUCCESS) {
        std::cerr << "Failed to set OpenCL kernel arguments: " << err << std::endl;
        return;
    }
   // std::cout << "OpenCL kernel arguments set successfully!" << std::endl;

    // Define optimal work size
    size_t globalWorkSize[2] = { (size_t)frame.cols, (size_t)frame.rows };

    // Use event-based execution to avoid blocking calls
    cl_event kernel_event;
    //std::cout << "Executing OpenCL kernel..." << std::endl;
    err = clEnqueueNDRangeKernel(queue, kernel, 2, nullptr, globalWorkSize, nullptr, 0, nullptr, &kernel_event);
    if (err != CL_SUCCESS) {
        std::cerr << "Failed to execute OpenCL kernel: " << err << std::endl;
        return;
    }
    //std::cout << "OpenCL kernel executed successfully!" << std::endl;

    // Wait for kernel execution to finish asynchronously
    clWaitForEvents(1, &kernel_event);
    clFinish(queue);
    //std::cout << "OpenCL kernel execution completed!" << std::endl;

    // Read back the processed image
    mappedPtr = (uchar*)clEnqueueMapBuffer(queue, buffer, CL_TRUE, CL_MAP_READ,
        0, contiguousFrame.total() * contiguousFrame.elemSize(),
        0, nullptr, nullptr, &err);
    if (err != CL_SUCCESS) {
        std::cerr << "Failed to map OpenCL buffer for reading: " << err << std::endl;
        return;
    }
    //std::cout << "OpenCL buffer mapped for reading successfully!" << std::endl;

    memcpy(contiguousFrame.data, mappedPtr, contiguousFrame.total() * contiguousFrame.elemSize());
    clEnqueueUnmapMemObject(queue, buffer, mappedPtr, 0, nullptr, nullptr);
   // std::cout << "Processed frame data copied back to host!" << std::endl;

    // Cleanup
    clReleaseMemObject(buffer);
    //std::cout << "OpenCL buffer released!" << std::endl;
}

// Process frames with CPU or GPU filtering
void process_frames(AVFormatContext* input_fmt_ctx, AVCodecContext* dec_ctx, AVCodecContext* enc_ctx, AVFormatContext* output_fmt_ctx, int video_stream_idx, bool use_opencl, cl_context opencl_context, cl_command_queue opencl_queue, cl_kernel opencl_kernel) {
    SwsContext* to_bgr_ctx = sws_getContext(dec_ctx->width, dec_ctx->height, dec_ctx->pix_fmt,
        dec_ctx->width, dec_ctx->height, AV_PIX_FMT_BGR24,
        SWS_BILINEAR, nullptr, nullptr, nullptr);

    SwsContext* from_bgr_ctx = sws_getContext(dec_ctx->width, dec_ctx->height, AV_PIX_FMT_BGR24,
        enc_ctx->width, enc_ctx->height, enc_ctx->pix_fmt,
        SWS_BILINEAR, nullptr, nullptr, nullptr);

    AVFrame* dec_frame = av_frame_alloc();
    AVFrame* bgr_frame = av_frame_alloc();
    AVFrame* enc_frame = av_frame_alloc();
    AVPacket* packet = av_packet_alloc();

    uint8_t* bgr_buffer = new uint8_t[av_image_get_buffer_size(AV_PIX_FMT_BGR24, dec_ctx->width, dec_ctx->height, 32)];
    av_image_fill_arrays(bgr_frame->data, bgr_frame->linesize, bgr_buffer,
        AV_PIX_FMT_BGR24, dec_ctx->width, dec_ctx->height, 32);

    enc_frame->format = enc_ctx->pix_fmt;
    enc_frame->width = enc_ctx->width;
    enc_frame->height = enc_ctx->height;
    av_frame_get_buffer(enc_frame, 32);

    int frame_count = 0;
    while (av_read_frame(input_fmt_ctx, packet) >= 0) {
        if (packet->stream_index == video_stream_idx) {
            avcodec_send_packet(dec_ctx, packet);
            while (avcodec_receive_frame(dec_ctx, dec_frame) == 0) {
                frame_count++;
               // std::cout << "Processing frame " << frame_count << std::endl;

                sws_scale(to_bgr_ctx, dec_frame->data, dec_frame->linesize, 0, dec_ctx->height,
                    bgr_frame->data, bgr_frame->linesize);

                cv::Mat cv_frame(dec_ctx->height, dec_ctx->width, CV_8UC3, bgr_frame->data[0], bgr_frame->linesize[0]);

                // Apply green filter (CPU or GPU)
                if (use_opencl) {
                    apply_green_filter_gpu(cv_frame, opencl_context, opencl_queue, opencl_kernel);
                }
                else {
                    apply_green_filter_cpu(cv_frame);
                }

                av_frame_make_writable(enc_frame);
                sws_scale(from_bgr_ctx, bgr_frame->data, bgr_frame->linesize, 0, enc_ctx->height,
                    enc_frame->data, enc_frame->linesize);

                enc_frame->pts = av_rescale_q(dec_frame->pts, input_fmt_ctx->streams[video_stream_idx]->time_base, enc_ctx->time_base);
                avcodec_send_frame(enc_ctx, enc_frame);

                while (avcodec_receive_packet(enc_ctx, packet) == 0) {
                    av_packet_rescale_ts(packet, enc_ctx->time_base, output_fmt_ctx->streams[0]->time_base);
                    av_interleaved_write_frame(output_fmt_ctx, packet);
                    av_packet_unref(packet);
                }
            }
        }
        av_packet_unref(packet);
    }

    // Flush encoder
    avcodec_send_frame(enc_ctx, nullptr);
    while (avcodec_receive_packet(enc_ctx, packet) == 0) {
        av_packet_rescale_ts(packet, enc_ctx->time_base, output_fmt_ctx->streams[0]->time_base);
        av_interleaved_write_frame(output_fmt_ctx, packet);
        av_packet_unref(packet);
    }

    // Cleanup
    delete[] bgr_buffer;
    av_frame_free(&dec_frame);
    av_frame_free(&bgr_frame);
    av_frame_free(&enc_frame);
    av_packet_free(&packet);
    sws_freeContext(to_bgr_ctx);
    sws_freeContext(from_bgr_ctx);
}







// Initialize decoder
bool init_decoder(AVFormatContext*& fmt_ctx, AVCodecContext*& dec_ctx, int& video_stream_idx) {
    // Open input file
    if (avformat_open_input(&fmt_ctx, INPUT_FILE, nullptr, nullptr) < 0) {
        std::cerr << "Failed to open input file: " << INPUT_FILE << std::endl;
        return false;
    }

    // Retrieve stream information
    if (avformat_find_stream_info(fmt_ctx, nullptr) < 0) {
        std::cerr << "Failed to retrieve stream information" << std::endl;
        return false;
    }

    // Find the video stream
    video_stream_idx = -1;
    for (int i = 0; i < fmt_ctx->nb_streams; ++i) {
        if (fmt_ctx->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_VIDEO) {
            video_stream_idx = i;
            break;
        }
    }
    if (video_stream_idx == -1) {
        std::cerr << "No video stream found in the input file" << std::endl;
        return false;
    }

    // Initialize decoder
    AVCodecParameters* codec_par = fmt_ctx->streams[video_stream_idx]->codecpar;
    const AVCodec* decoder = avcodec_find_decoder(codec_par->codec_id);
    if (!decoder) {
        std::cerr << "Unsupported codec" << std::endl;
        return false;
    }

    dec_ctx = avcodec_alloc_context3(decoder);
    avcodec_parameters_to_context(dec_ctx, codec_par);
    if (avcodec_open2(dec_ctx, decoder, nullptr) < 0) {
        std::cerr << "Failed to open decoder" << std::endl;
        return false;
    }

    return true;
}

// Initialize encoder
bool init_encoder(AVFormatContext*& fmt_ctx, AVCodecContext*& enc_ctx, AVStream*& out_stream, AVRational input_time_base, int width, int height, const char* output_file) {
    avformat_alloc_output_context2(&fmt_ctx, nullptr, nullptr, output_file);
    const AVCodec* encoder = avcodec_find_encoder(AV_CODEC_ID_H264);
    enc_ctx = avcodec_alloc_context3(encoder);

    // Set encoder parameters
    enc_ctx->width = width;
    enc_ctx->height = height;
    enc_ctx->pix_fmt = AV_PIX_FMT_YUV420P;
    enc_ctx->time_base = input_time_base;

    av_opt_set(enc_ctx->priv_data, "crf", "23", 0);
    av_opt_set(enc_ctx->priv_data, "preset", "medium", 0);
    av_opt_set(enc_ctx->priv_data, "tune", "film", 0);

    if (avcodec_open2(enc_ctx, encoder, nullptr) < 0) {
        std::cerr << "Could not open encoder" << std::endl;
        return false;
    }

    out_stream = avformat_new_stream(fmt_ctx, encoder);
    avcodec_parameters_from_context(out_stream->codecpar, enc_ctx);
    return avio_open(&fmt_ctx->pb, output_file, AVIO_FLAG_WRITE) >= 0;
}

int main() {
    avformat_network_init();

    AVFormatContext* input_fmt_ctx = nullptr;
    AVCodecContext* dec_ctx = nullptr;
    AVFormatContext* output_cpu_fmt_ctx = nullptr;
    AVCodecContext* cpu_enc_ctx = nullptr;
    AVStream* cpu_out_stream = nullptr;
    int video_stream_idx = -1;

    // Initialize decoder
    if (!init_decoder(input_fmt_ctx, dec_ctx, video_stream_idx)) {
        std::cerr << "Decoder initialization failed" << std::endl;
        return -1;
    }

    // Initialize CPU encoder
    AVRational input_time_base = input_fmt_ctx->streams[video_stream_idx]->time_base;
    if (!init_encoder(output_cpu_fmt_ctx, cpu_enc_ctx, cpu_out_stream, input_time_base, dec_ctx->width, dec_ctx->height, OUTPUT_CPU_FILE)) {
        std::cerr << "CPU Encoder initialization failed" << std::endl;
        return -1;
    }

    // Write header & process frames with CPU filter
    avformat_write_header(output_cpu_fmt_ctx, nullptr);
    auto start_cpu = std::chrono::high_resolution_clock::now();
    process_frames(input_fmt_ctx, dec_ctx, cpu_enc_ctx, output_cpu_fmt_ctx, video_stream_idx, false, nullptr, nullptr, nullptr);
    auto end_cpu = std::chrono::high_resolution_clock::now();
    av_write_trailer(output_cpu_fmt_ctx);

    // Reinitialize decoder for GPU processing
    av_seek_frame(input_fmt_ctx, video_stream_idx, 0, AVSEEK_FLAG_BACKWARD);
    avcodec_flush_buffers(dec_ctx);

    // Initialize GPU encoder
    AVFormatContext* output_gpu_fmt_ctx = nullptr;
    AVCodecContext* gpu_enc_ctx = nullptr;
    AVStream* gpu_out_stream = nullptr;
    if (!init_encoder(output_gpu_fmt_ctx, gpu_enc_ctx, gpu_out_stream, input_time_base, dec_ctx->width, dec_ctx->height, OUTPUT_GPU_FILE)) {
        std::cerr << "GPU Encoder initialization failed" << std::endl;
        return -1;
    }

    // Initialize OpenCL
    cl_device_id opencl_device = nullptr;
    cl_command_queue opencl_queue = nullptr;
    cl_kernel opencl_kernel = nullptr;
    cl_context opencl_context = init_opencl(opencl_device, opencl_queue, opencl_kernel);
    if (!opencl_context) {
        std::cerr << "OpenCL initialization failed" << std::endl;
        return -1;
    }

    // Write header & process frames with GPU filter
    avformat_write_header(output_gpu_fmt_ctx, nullptr);
    auto start_gpu = std::chrono::high_resolution_clock::now();
    process_frames(input_fmt_ctx, dec_ctx, gpu_enc_ctx, output_gpu_fmt_ctx, video_stream_idx, true, opencl_context, opencl_queue, opencl_kernel);
    auto end_gpu = std::chrono::high_resolution_clock::now();
    av_write_trailer(output_gpu_fmt_ctx);

    // Measure execution times
    std::chrono::duration<double> cpu_time = end_cpu - start_cpu;
    std::chrono::duration<double> gpu_time = end_gpu - start_gpu;

    std::cout << "CPU Processing Time: " << cpu_time.count() << " seconds" << std::endl;
    std::cout << "GPU Processing Time: " << gpu_time.count() << " seconds" << std::endl;

    // Cleanup
    avcodec_free_context(&dec_ctx);
    avcodec_free_context(&cpu_enc_ctx);
    avcodec_free_context(&gpu_enc_ctx);
    avformat_close_input(&input_fmt_ctx);
    avformat_free_context(output_cpu_fmt_ctx);
    avformat_free_context(output_gpu_fmt_ctx);

    if (opencl_context) {
        clReleaseKernel(opencl_kernel);
        clReleaseCommandQueue(opencl_queue);
        clReleaseContext(opencl_context);
    }

    return 0;
}
#include <opencv2/opencv.hpp>
extern "C" {
#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
#include <libswscale/swscale.h>
#include <libavutil/imgutils.h>  // ✅ Required for av_image_fill_arrays

}

int main() {
    avformat_network_init();

    // FFmpeg setup
    AVFormatContext* format_ctx = avformat_alloc_context();
    if (avformat_open_input(&format_ctx, "input.mp4", nullptr, nullptr) < 0) {
        return -1;
    }
    avformat_find_stream_info(format_ctx, nullptr);

    // Find video stream
    int video_stream_index = -1;
    for (unsigned i = 0; i < format_ctx->nb_streams; ++i) {
        if (format_ctx->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_VIDEO) {
            video_stream_index = i;
            break;
        }
    }

    // Get codec context
    AVCodecParameters* codec_par = format_ctx->streams[video_stream_index]->codecpar;
    const AVCodec* codec = avcodec_find_decoder(codec_par->codec_id);
    AVCodecContext* codec_ctx = avcodec_alloc_context3(codec);
    avcodec_parameters_to_context(codec_ctx, codec_par);
    avcodec_open2(codec_ctx, codec, nullptr);

    // Frame and conversion setup
    AVFrame* frame = av_frame_alloc();
    AVFrame* rgb_frame = av_frame_alloc();
    SwsContext* sws_ctx = sws_getContext(
        codec_ctx->width, codec_ctx->height, codec_ctx->pix_fmt,
        codec_ctx->width, codec_ctx->height, AV_PIX_FMT_BGR24,
        SWS_BILINEAR, nullptr, nullptr, nullptr
    );
    uint8_t* buffer = new uint8_t[av_image_get_buffer_size(AV_PIX_FMT_BGR24, codec_ctx->width, codec_ctx->height, 1)];
    av_image_fill_arrays(rgb_frame->data, rgb_frame->linesize, buffer, AV_PIX_FMT_BGR24, codec_ctx->width, codec_ctx->height, 1);

    // OpenCV window
    cv::namedWindow("Video", cv::WINDOW_AUTOSIZE);

    // Read and display frames
    AVPacket* packet = av_packet_alloc();
    while (av_read_frame(format_ctx, packet) >= 0) {
        if (packet->stream_index == video_stream_index) {
            avcodec_send_packet(codec_ctx, packet);
            while (avcodec_receive_frame(codec_ctx, frame) == 0) {
                // Convert to BGR for OpenCV
                sws_scale(sws_ctx, frame->data, frame->linesize, 0, codec_ctx->height, rgb_frame->data, rgb_frame->linesize);

                // Create OpenCV matrix
                cv::Mat image(codec_ctx->height, codec_ctx->width, CV_8UC3, rgb_frame->data[0], rgb_frame->linesize[0]);

                // Display frame
                cv::imshow("Video", image);
                if (cv::waitKey(30) >= 0) break;
            }
        }
        av_packet_unref(packet);
    }

    // Cleanup
    cv::destroyAllWindows();
    avformat_close_input(&format_ctx);
    avcodec_free_context(&codec_ctx);
    av_frame_free(&frame);
    av_frame_free(&rgb_frame);
    sws_freeContext(sws_ctx);
    delete[] buffer;
    return 0;
}
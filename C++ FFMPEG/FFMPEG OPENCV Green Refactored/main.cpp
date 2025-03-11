#include <opencv2/opencv.hpp>

extern "C" {
#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
#include <libavutil/avutil.h>
#include <libavutil/imgutils.h>
#include <libavutil/opt.h>
#include <libswscale/swscale.h>
}

const char* INPUT_FILE = "input.mp4";
const char* OUTPUT_FILE = "output.mp4";

// Apply green filter using OpenCV
void apply_green_filter(cv::Mat& frame) {
    std::vector<cv::Mat> channels;
    cv::split(frame, channels);
    channels[0] = cv::Scalar(0);  // Zero out Blue
    channels[2] = cv::Scalar(0);  // Zero out Red
    cv::merge(channels, frame);
}

// Initialize FFmpeg decoder
bool init_decoder(AVFormatContext*& fmt_ctx, AVCodecContext*& dec_ctx, int& video_stream_idx) {
    if (avformat_open_input(&fmt_ctx, INPUT_FILE, nullptr, nullptr) < 0) return false;
    if (avformat_find_stream_info(fmt_ctx, nullptr) < 0) return false;

    // Find video stream
    video_stream_idx = -1;
    for (int i = 0; i < fmt_ctx->nb_streams; ++i) {
        if (fmt_ctx->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_VIDEO) {
            video_stream_idx = i;
            break;
        }
    }
    if (video_stream_idx == -1) return false;

    // Initialize decoder
    AVCodecParameters* codec_par = fmt_ctx->streams[video_stream_idx]->codecpar;
    const AVCodec* decoder = avcodec_find_decoder(codec_par->codec_id);
    dec_ctx = avcodec_alloc_context3(decoder);
    avcodec_parameters_to_context(dec_ctx, codec_par);
    return avcodec_open2(dec_ctx, decoder, nullptr) >= 0;
}

// Initialize FFmpeg encoder
bool init_encoder(AVFormatContext*& fmt_ctx, AVCodecContext*& enc_ctx, AVStream*& out_stream, AVRational input_time_base, int width, int height) {
    avformat_alloc_output_context2(&fmt_ctx, nullptr, nullptr, OUTPUT_FILE);
    const AVCodec* encoder = avcodec_find_encoder(AV_CODEC_ID_H264);
    enc_ctx = avcodec_alloc_context3(encoder);

    // ✅ Ensure dimensions are set correctly
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
    return avio_open(&fmt_ctx->pb, OUTPUT_FILE, AVIO_FLAG_WRITE) >= 0;
}

// Process frames: decode, apply filter, encode
void process_frames(AVFormatContext* input_fmt_ctx, AVCodecContext* dec_ctx, AVCodecContext* enc_ctx, AVFormatContext* output_fmt_ctx, int video_stream_idx) {
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

    while (av_read_frame(input_fmt_ctx, packet) >= 0) {
        if (packet->stream_index == video_stream_idx) {
            avcodec_send_packet(dec_ctx, packet);
            while (avcodec_receive_frame(dec_ctx, dec_frame) == 0) {
                sws_scale(to_bgr_ctx, dec_frame->data, dec_frame->linesize, 0, dec_ctx->height,
                    bgr_frame->data, bgr_frame->linesize);

                cv::Mat cv_frame(dec_ctx->height, dec_ctx->width, CV_8UC3, bgr_frame->data[0], bgr_frame->linesize[0]);
                apply_green_filter(cv_frame);

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

int main() {
    avformat_network_init();

    AVFormatContext* input_fmt_ctx = nullptr;
    AVCodecContext* dec_ctx = nullptr;
    AVFormatContext* output_fmt_ctx = nullptr;
    AVCodecContext* enc_ctx = nullptr;
    AVStream* out_stream = nullptr;
    int video_stream_idx = -1;

    // Initialize decoder
    if (!init_decoder(input_fmt_ctx, dec_ctx, video_stream_idx)) {
        std::cerr << "Decoder initialization failed" << std::endl;
        return -1;
    }

    // Initialize encoder
    AVRational input_time_base = input_fmt_ctx->streams[video_stream_idx]->time_base;
    if (!init_encoder(output_fmt_ctx, enc_ctx, out_stream, input_time_base, dec_ctx->width, dec_ctx->height)) {
        std::cerr << "Encoder initialization failed" << std::endl;
        return -1;
    }

    // Write header & process frames
    avformat_write_header(output_fmt_ctx, nullptr);
    process_frames(input_fmt_ctx, dec_ctx, enc_ctx, output_fmt_ctx, video_stream_idx);
    av_write_trailer(output_fmt_ctx);

    // Cleanup
    avcodec_free_context(&dec_ctx);
    avcodec_free_context(&enc_ctx);
    avformat_close_input(&input_fmt_ctx);
    avformat_free_context(output_fmt_ctx);

    return 0;
}

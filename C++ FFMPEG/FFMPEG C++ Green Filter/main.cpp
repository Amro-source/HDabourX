extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/imgutils.h>
#include <libswscale/swscale.h>
}

#include <iostream>

void apply_green_filter(uint8_t* rgb_data, int width, int height, int linesize) {
    for (int y = 0; y < height; ++y) {
        uint8_t* row = rgb_data + y * linesize;
        for (int x = 0; x < width * 3; x += 3) {
            row[x] = 0;  // Blue channel
            row[x + 2] = 0;  // Red channel
        }
    }
}

int main() {
    avformat_network_init();

    const char* input_filename = "input.mp4";
    const char* output_filename = "output.mp4";

    AVFormatContext* input_format_ctx = nullptr;
    AVFormatContext* output_format_ctx = nullptr;
    AVCodecContext* decode_ctx = nullptr;
    AVCodecContext* encode_ctx = nullptr;
    SwsContext* sws_ctx_to_rgb = nullptr;
    SwsContext* sws_ctx_from_rgb = nullptr;

    try {
        if (avformat_open_input(&input_format_ctx, input_filename, nullptr, nullptr) < 0)
            throw std::runtime_error("Could not open input");
        if (avformat_find_stream_info(input_format_ctx, nullptr) < 0)
            throw std::runtime_error("Could not find stream info");

        int video_stream_index = -1;
        for (unsigned i = 0; i < input_format_ctx->nb_streams; ++i) {
            if (input_format_ctx->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_VIDEO) {
                video_stream_index = i;
                break;
            }
        }
        if (video_stream_index == -1) throw std::runtime_error("No video stream found");

        AVCodecParameters* codec_par = input_format_ctx->streams[video_stream_index]->codecpar;
        const AVCodec* decoder = avcodec_find_decoder(codec_par->codec_id);
        if (!decoder) throw std::runtime_error("Decoder not found");

        decode_ctx = avcodec_alloc_context3(decoder);
        if (!decode_ctx) throw std::runtime_error("Could not allocate decode context");

        if (avcodec_parameters_to_context(decode_ctx, codec_par) < 0)
            throw std::runtime_error("Could not copy codec parameters");
        if (avcodec_open2(decode_ctx, decoder, nullptr) < 0)
            throw std::runtime_error("Could not open decoder");

        // Create output format context
        avformat_alloc_output_context2(&output_format_ctx, nullptr, nullptr, output_filename);
        if (!output_format_ctx) throw std::runtime_error("Could not create output context");

        const AVCodec* encoder = avcodec_find_encoder(codec_par->codec_id);
        if (!encoder) throw std::runtime_error("Encoder not found");

        encode_ctx = avcodec_alloc_context3(encoder);
        if (!encode_ctx) throw std::runtime_error("Could not allocate encode context");

        // Set encoding parameters
        encode_ctx->bit_rate = decode_ctx->bit_rate;
        encode_ctx->width = decode_ctx->width;
        encode_ctx->height = decode_ctx->height;
        encode_ctx->pix_fmt = encoder->pix_fmts ? encoder->pix_fmts[0] : decode_ctx->pix_fmt;
        encode_ctx->time_base = av_inv_q(decode_ctx->framerate);

        if (avcodec_open2(encode_ctx, encoder, nullptr) < 0)
            throw std::runtime_error("Could not open encoder");

        AVStream* out_stream = avformat_new_stream(output_format_ctx, encoder);
        if (!out_stream) throw std::runtime_error("Could not create output stream");

        avcodec_parameters_from_context(out_stream->codecpar, encode_ctx);

        if (!(output_format_ctx->oformat->flags & AVFMT_NOFILE)) {
            if (avio_open(&output_format_ctx->pb, output_filename, AVIO_FLAG_WRITE) < 0) {
                throw std::runtime_error("Could not open output file");
            }
        }
        avformat_write_header(output_format_ctx, nullptr);

        AVFrame* frame = av_frame_alloc();
        AVFrame* rgb_frame = av_frame_alloc();
        AVFrame* converted_frame = av_frame_alloc();
        AVPacket* packet = av_packet_alloc();
        if (!frame || !rgb_frame || !converted_frame || !packet)
            throw std::runtime_error("Memory allocation failed");

        int num_bytes = av_image_get_buffer_size(AV_PIX_FMT_RGB24, decode_ctx->width, decode_ctx->height, 32);
        uint8_t* rgb_buffer = (uint8_t*)av_malloc(num_bytes);
        if (!rgb_buffer) throw std::runtime_error("Could not allocate RGB buffer");

        av_image_fill_arrays(rgb_frame->data, rgb_frame->linesize, rgb_buffer,
            AV_PIX_FMT_RGB24, decode_ctx->width, decode_ctx->height, 32);

        converted_frame->format = encode_ctx->pix_fmt;
        converted_frame->width = encode_ctx->width;
        converted_frame->height = encode_ctx->height;
        if (av_frame_get_buffer(converted_frame, 32) < 0)
            throw std::runtime_error("Could not allocate frame buffer");

        sws_ctx_to_rgb = sws_getContext(decode_ctx->width, decode_ctx->height, decode_ctx->pix_fmt,
            decode_ctx->width, decode_ctx->height, AV_PIX_FMT_RGB24,
            SWS_BILINEAR, nullptr, nullptr, nullptr);
        sws_ctx_from_rgb = sws_getContext(decode_ctx->width, decode_ctx->height, AV_PIX_FMT_RGB24,
            encode_ctx->width, encode_ctx->height, encode_ctx->pix_fmt,
            SWS_BILINEAR, nullptr, nullptr, nullptr);
        if (!sws_ctx_to_rgb || !sws_ctx_from_rgb)
            throw std::runtime_error("Could not create scaling contexts");

        while (av_read_frame(input_format_ctx, packet) >= 0) {
            if (packet->stream_index == video_stream_index) {
                if (avcodec_send_packet(decode_ctx, packet) < 0)
                    throw std::runtime_error("Error sending packet to decoder");

                while (avcodec_receive_frame(decode_ctx, frame) == 0) {
                    sws_scale(sws_ctx_to_rgb, frame->data, frame->linesize, 0, decode_ctx->height,
                        rgb_frame->data, rgb_frame->linesize);

                    apply_green_filter(rgb_frame->data[0], decode_ctx->width, decode_ctx->height, rgb_frame->linesize[0]);

                    sws_scale(sws_ctx_from_rgb, rgb_frame->data, rgb_frame->linesize, 0, encode_ctx->height,
                        converted_frame->data, converted_frame->linesize);

                    converted_frame->pts = frame->pts;

                    if (avcodec_send_frame(encode_ctx, converted_frame) < 0)
                        throw std::runtime_error("Error sending frame to encoder");

                    while (avcodec_receive_packet(encode_ctx, packet) == 0) {
                        av_interleaved_write_frame(output_format_ctx, packet);
                        av_packet_unref(packet);
                    }
                }
            }
            av_packet_unref(packet);
        }

        av_write_trailer(output_format_ctx);

        av_free(rgb_buffer);
        avcodec_free_context(&decode_ctx);
        avcodec_free_context(&encode_ctx);
        avformat_close_input(&input_format_ctx);
        avformat_free_context(output_format_ctx);
        av_frame_free(&frame);
        av_frame_free(&rgb_frame);
        av_frame_free(&converted_frame);
        av_packet_free(&packet);
        sws_freeContext(sws_ctx_to_rgb);
        sws_freeContext(sws_ctx_from_rgb);

        return 0;

    }
    catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return -1;
    }
}

#include <zlib.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>

#define CHUNK 16384

int compress(const char* input, unsigned char** output, unsigned long* output_length) {
    z_stream stream;
    int err;
    unsigned char out[CHUNK];

    stream.zalloc = Z_NULL;
    stream.zfree = Z_NULL;
    stream.opaque = Z_NULL;

    err = deflateInit(&stream, Z_DEFAULT_COMPRESSION);
    if (err != Z_OK) {
        return err;
    }

    stream.avail_in = (unsigned int)strlen(input);
    stream.next_in = (unsigned char*)input;

    *output_length = 0;
    while (stream.avail_in > 0) {
        stream.avail_out = CHUNK;
        stream.next_out = out;
        err = deflate(&stream, Z_FINISH);
        if (err != Z_STREAM_END) {
            deflateEnd(&stream);
            return err;
        }
        *output_length += CHUNK - stream.avail_out;
    }

    *output = (unsigned char*)malloc(*output_length);
    memcpy(*output, out, *output_length);

    deflateEnd(&stream);
    return Z_OK;
}

int decompress(const unsigned char* input, unsigned long input_length, char** output, unsigned long* output_length) {
    z_stream stream;
    int err;
    unsigned char out[CHUNK];

    stream.zalloc = Z_NULL;
    stream.zfree = Z_NULL;
    stream.opaque = Z_NULL;

    err = inflateInit(&stream);
    if (err != Z_OK) {
        return err;
    }

    stream.avail_in = (unsigned int)input_length;
    stream.next_in = (unsigned char*)input;

    *output_length = 0;
    while (stream.avail_in > 0) {
        stream.avail_out = CHUNK;
        stream.next_out = out;
        err = inflate(&stream, Z_NO_FLUSH);
        if (err != Z_OK && err != Z_STREAM_END) {
            inflateEnd(&stream);
            return err;
        }
        *output_length += CHUNK - stream.avail_out;
    }

    *output = (char*)malloc(*output_length + 1);
    memcpy(*output, out, *output_length);
    (*output)[*output_length] = '\0';

    inflateEnd(&stream);
    return Z_OK;
}

int main() {
    const char* input = "Hello, World!";
    unsigned char* compressed;
    unsigned long compressed_length;
    char* decompressed;
    unsigned long decompressed_length;

    int err = compress(input, &compressed, &compressed_length);
    if (err != Z_OK) {
        printf("Compression failed: %d\n", err);
        return 1;
    }

    printf("Compressed length: %lu\n", compressed_length);

    err = decompress(compressed, compressed_length, &decompressed, &decompressed_length);
    if (err != Z_OK) {
        printf("Decompression failed: %d\n", err);
        return 1;
    }

    printf("Decompressed: %s\n", decompressed);

    free(compressed);
    free(decompressed);

    return 0;
}

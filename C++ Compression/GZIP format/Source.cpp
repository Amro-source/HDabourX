#include <zlib.h>
#include <iostream>
#include <vector>
#include <iomanip>
#include <cstring>

int main() {
    // Original data
    const char* input = "Hello, GZIP! This is a test of GZIP compression using zlib.";
    size_t input_size = std::strlen(input);
    std::cout << "Original size: " << input_size << " bytes" << std::endl;

    // Compression
    z_stream comp_strm = {};
    comp_strm.zalloc = Z_NULL;
    comp_strm.zfree = Z_NULL;
    comp_strm.opaque = Z_NULL;

    // Initialize GZIP compression
    int ret = deflateInit2(
        &comp_strm,
        Z_DEFAULT_COMPRESSION,
        Z_DEFLATED,
        31, // 15 (window size) + 16 (GZIP format)
        8,
        Z_DEFAULT_STRATEGY
    );
    if (ret != Z_OK) {
        std::cerr << "Compression init failed: " << ret << std::endl;
        return 1;
    }

    // Calculate proper buffer size for GZIP
    uLongf compressed_size_max = deflateBound(&comp_strm, input_size);
    std::vector<Bytef> compressed_buffer(compressed_size_max);
    std::cout << "Allocated compression buffer: " << compressed_size_max << " bytes" << std::endl;

    // Compress data
    comp_strm.avail_in = input_size;
    comp_strm.next_in = reinterpret_cast<Bytef*>(const_cast<char*>(input));
    comp_strm.avail_out = compressed_size_max;
    comp_strm.next_out = compressed_buffer.data();

    ret = deflate(&comp_strm, Z_FINISH);
    if (ret != Z_STREAM_END) {
        std::cerr << "Compression failed: " << ret << std::endl;
        deflateEnd(&comp_strm);
        return 1;
    }
    size_t compressed_size = compressed_size_max - comp_strm.avail_out;
    deflateEnd(&comp_strm);
    std::cout << "Compression succeeded. Compressed size: " << compressed_size << " bytes" << std::endl;

    // Decompression
    z_stream decomp_strm = {};
    decomp_strm.zalloc = Z_NULL;
    decomp_strm.zfree = Z_NULL;
    decomp_strm.opaque = Z_NULL;

    // Initialize GZIP decompression
    ret = inflateInit2(&decomp_strm, 31); // 31 = GZIP format
    if (ret != Z_OK) {
        std::cerr << "Decompression init failed: " << ret << std::endl;
        return 1;
    }

    // Allocate decompression buffer
    std::vector<Bytef> decompressed_buffer(input_size);
    decomp_strm.avail_in = compressed_size;
    decomp_strm.next_in = compressed_buffer.data();
    decomp_strm.avail_out = input_size;
    decomp_strm.next_out = decompressed_buffer.data();

    ret = inflate(&decomp_strm, Z_FINISH);
    if (ret != Z_STREAM_END) {
        std::cerr << "Decompression failed: " << ret << std::endl;
        inflateEnd(&decomp_strm);
        return 1;
    }
    size_t decompressed_size = input_size - decomp_strm.avail_out;
    inflateEnd(&decomp_strm);

    // Verify results
    bool success = (
        decompressed_size == input_size &&
        memcmp(input, decompressed_buffer.data(), input_size) == 0
        );

    // Final output
    std::cout << "Decompressed size: " << decompressed_size << " bytes" << std::endl;
    std::cout << "Verification: " << (success ? "SUCCESS" : "FAILED") << std::endl;

    return 0;
}
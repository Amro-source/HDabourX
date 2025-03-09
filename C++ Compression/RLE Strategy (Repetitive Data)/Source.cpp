#include <zlib.h>
#include <iostream>
#include <vector>
#include <iomanip>
#include <cstring>

int main() {
    // Create highly repetitive data (ideal for RLE)
    const char* input = "AAAAAAAAAABBBBBBBBBBCCCCCCCCCCDDDDDDDDDD";
    uLong input_size = std::strlen(input);

    // Compression parameters
    z_stream comp_strm = {};
    comp_strm.zalloc = Z_NULL;
    comp_strm.zfree = Z_NULL;
    comp_strm.opaque = Z_NULL;

    // Initialize with RLE strategy
    int ret = deflateInit2(
        &comp_strm,
        Z_DEFAULT_COMPRESSION, // Compression level (RLE strategy handles repetition)
        Z_DEFLATED,            // Compression method
        15,                    // Window bits (32KB)
        8,                     // Memory level
        Z_RLE                  // RLE strategy for repetitive data
    );
    if (ret != Z_OK) {
        std::cerr << "Compression init failed: " << ret << std::endl;
        return 1;
    }

    // Allocate compression buffer
    uLongf compressed_size_max = deflateBound(&comp_strm, input_size);
    std::vector<Bytef> compressed(compressed_size_max);

    // Perform compression
    comp_strm.avail_in = input_size;
    comp_strm.next_in = (Bytef*)input;
    comp_strm.avail_out = compressed_size_max;
    comp_strm.next_out = compressed.data();

    ret = deflate(&comp_strm, Z_FINISH);
    if (ret != Z_STREAM_END) {
        std::cerr << "Compression failed: " << ret << std::endl;
        deflateEnd(&comp_strm);
        return 1;
    }
    uLongf compressed_size = compressed_size_max - comp_strm.avail_out;
    deflateEnd(&comp_strm);

    // Decompression
    z_stream decomp_strm = {};
    decomp_strm.zalloc = Z_NULL;
    decomp_strm.zfree = Z_NULL;
    decomp_strm.opaque = Z_NULL;

    ret = inflateInit(&decomp_strm);
    if (ret != Z_OK) {
        std::cerr << "Decompression init failed: " << ret << std::endl;
        return 1;
    }

    std::vector<Bytef> decompressed(input_size);
    decomp_strm.avail_in = compressed_size;
    decomp_strm.next_in = compressed.data();
    decomp_strm.avail_out = input_size;
    decomp_strm.next_out = decompressed.data();

    ret = inflate(&decomp_strm, Z_FINISH);
    if (ret != Z_STREAM_END) {
        std::cerr << "Decompression failed: " << ret << std::endl;
        inflateEnd(&decomp_strm);
        return 1;
    }
    inflateEnd(&decomp_strm);

    // Verify results
    bool success = (
        decomp_strm.total_out == input_size &&
        memcmp(input, decompressed.data(), input_size) == 0
        );

    // Output results
    std::cout << "Original size: " << input_size << " bytes" << std::endl;
    std::cout << "Compressed size: " << compressed_size << " bytes" << std::endl;
    std::cout << "Compression ratio: "
        << std::fixed << std::setprecision(1)
        << (1.0 - (compressed_size / (double)input_size)) * 100
        << "%" << std::endl;
    std::cout << "Verification: " << (success ? "SUCCESS" : "FAILED") << std::endl;

    return 0;
}
#include <zlib.h>
#include <iostream>
#include <vector>
#include <iomanip>

int main() {
    // Input data
    const char* input = "Hello, ZLIB compression!";
    size_t input_size = std::strlen(input);

    // Initialize zlib stream
    z_stream strm;
    strm.zalloc = Z_NULL;
    strm.zfree = Z_NULL;
    strm.opaque = Z_NULL;

    // Initialize deflation (ZLIB format)
    int ret = deflateInit(&strm, Z_DEFAULT_COMPRESSION);
    if (ret != Z_OK) {
        std::cerr << "deflateInit failed: " << ret << std::endl;
        return 1;
    }

    // Prepare input
    strm.avail_in = static_cast<uInt>(input_size);
    strm.next_in = reinterpret_cast<Bytef*>(const_cast<char*>(input));

    // Prepare output buffer
    std::vector<Bytef> output_buffer(1024); // 1KB buffer
    strm.avail_out = static_cast<uInt>(output_buffer.size());
    strm.next_out = output_buffer.data();

    // Perform compression
    ret = deflate(&strm, Z_FINISH);
    if (ret != Z_STREAM_END && ret != Z_OK) {
        std::cerr << "deflation failed: " << ret << std::endl;
        deflateEnd(&strm);
        return 1;
    }

    // Get compressed size
    size_t compressed_size = output_buffer.size() - strm.avail_out;

    // Clean up
    deflateEnd(&strm);

    // Output results
    std::cout << "Original size: " << input_size << " bytes" << std::endl;
    std::cout << "Compressed size: " << compressed_size << " bytes" << std::endl;
    std::cout << "Compression ratio: "
        << std::fixed << std::setprecision(1)
        << (1.0 - (static_cast<float>(compressed_size) / input_size)) * 100
        << "%" << std::endl;

    // Show compressed data in hex (first 10 bytes)
    std::cout << "Compressed data (hex): ";
    for (size_t i = 0; i < std::min(compressed_size, size_t(10)); ++i) {
        std::cout << std::hex << std::setw(2) << std::setfill('0')
            << static_cast<int>(output_buffer[i]) << " ";
    }
    std::cout << std::endl;

    return 0;
}
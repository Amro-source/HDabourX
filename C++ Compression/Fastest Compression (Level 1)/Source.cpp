#include <zlib.h>
#include <iostream>
#include <vector>
#include <cstring>

int main() {
    const char* data = "Hello, Level 1 Compression!";
    uLong data_len = std::strlen(data);

    // Compression
    z_stream comp_strm = {};
    comp_strm.zalloc = Z_NULL;
    comp_strm.zfree = Z_NULL;
    comp_strm.opaque = Z_NULL;

    // Initialize with Level 1 compression
    int ret = deflateInit2(
        &comp_strm,
        1,                // Compression level 1 (fastest)
        Z_DEFLATED,       // Compression method
        15,               // Window bits (32KB)
        8,                // Memory level
        Z_DEFAULT_STRATEGY
    );
    if (ret != Z_OK) {
        std::cerr << "Compression init failed: " << ret << std::endl;
        return 1;
    }

    // Compress
    uLongf compressed_size = deflateBound(&comp_strm, data_len);
    std::vector<Bytef> compressed(compressed_size);
    comp_strm.avail_in = data_len;
    comp_strm.next_in = (Bytef*)data;
    comp_strm.avail_out = compressed_size;
    comp_strm.next_out = compressed.data();

    ret = deflate(&comp_strm, Z_FINISH);
    if (ret != Z_STREAM_END) {
        std::cerr << "Compression failed: " << ret << std::endl;
        deflateEnd(&comp_strm);
        return 1;
    }
    compressed_size = compressed_size - comp_strm.avail_out;
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

    std::vector<Bytef> decompressed(data_len);
    decomp_strm.avail_in = compressed_size;
    decomp_strm.next_in = compressed.data();
    decomp_strm.avail_out = data_len;
    decomp_strm.next_out = decompressed.data();

    ret = inflate(&decomp_strm, Z_FINISH);
    if (ret != Z_STREAM_END) {
        std::cerr << "Decompression failed: " << ret << std::endl;
        inflateEnd(&decomp_strm);
        return 1;
    }
    inflateEnd(&decomp_strm);

    // Verify
    bool success = (std::memcmp(data, decompressed.data(), data_len) == 0);
    std::cout << "Verification: " << (success ? "SUCCESS" : "FAILED") << std::endl;
    return 0;
}
#include <zlib.h>
#include <iostream>
#include <vector>
#include <cstring>

int main() {
    const char* data = "Hello, Level 9 Compression!";
    uLong data_len = std::strlen(data);

    // Compression
    z_stream comp_strm = {};
    comp_strm.zalloc = Z_NULL;
    comp_strm.zfree = Z_NULL;
    comp_strm.opaque = Z_NULL;

    // Initialize with Level 9 compression
    int ret = deflateInit2(
        &comp_strm,
        9,                // Compression level 9 (maximum)
        Z_DEFLATED,       // Compression method
        15,               // Window bits (32KB)
        9,                // Memory level (higher for better compression)
        Z_DEFAULT_STRATEGY
    );
    if (ret != Z_OK) {
        std::cerr << "Compression init failed: " << ret << std::endl;
        return 1;
    }

    // Compress (rest same as Level 1 example)
    // ... [same compression/decompression code as above] ...

    return 0;
}
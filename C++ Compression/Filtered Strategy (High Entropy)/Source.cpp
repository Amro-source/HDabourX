#include <zlib.h>
#include <iostream>
#include <vector>
#include <cstring>

int main() {
    // High-entropy data (e.g., pre-compressed)
    const char* data = "\x00\x01\x02\x03\x04\x05\x06\x07\x08\x09\x0a\x0b\x0c\x0d\x0e\x0f";
    uLong data_len = 16;

    // Compression
    z_stream comp_strm = {};
    comp_strm.zalloc = Z_NULL;
    comp_strm.zfree = Z_NULL;
    comp_strm.opaque = Z_NULL;

    // Initialize with FILTERED strategy
    int ret = deflateInit2(
        &comp_strm,
        Z_DEFAULT_COMPRESSION,
        Z_DEFLATED,
        15,               // Window bits
        8,                // Memory level
        Z_FILTERED        // Strategy for high-entropy data
    );
    if (ret != Z_OK) {
        std::cerr << "Compression init failed: " << ret << std::endl;
        return 1;
    }

    // Compress (rest same as previous examples)
    // ... [same compression/decompression code] ...

    return 0;
}
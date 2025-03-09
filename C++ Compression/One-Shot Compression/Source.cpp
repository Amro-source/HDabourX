#include <zlib.h>      // For zlib compression functions
#include <iostream>    // For std::cout
#include <cstring>     // Added for strlen()
#include <vector>      // Added for std::vector

int main() {
    const char* data = "Hello, World!";
    uLong data_len = std::strlen(data);  // Use uLong for zlib compatibility

    // Calculate required buffer size
    uLongf compressed_size = compressBound(data_len);
    std::vector<Bytef> compressed(compressed_size);

    // Perform compression
    int result = compress(
        compressed.data(),     // Destination buffer
        &compressed_size,      // Input: max size, Output: actual size
        (const Bytef*)data,    // Source data
        data_len               // Source data length
    );

    // Check for errors
    if (result != Z_OK) {
        std::cerr << "Compression failed with error code: " << result << std::endl;
        return 1;
    }

    // Show results
    std::cout << "Original size: " << data_len << " bytes" << std::endl;
    std::cout << "Compressed size: " << compressed_size << " bytes" << std::endl;
    std::cout << "Compression ratio: "
        << static_cast<float>(data_len - compressed_size) / data_len * 100
        << "%" << std::endl;

    return 0;
}
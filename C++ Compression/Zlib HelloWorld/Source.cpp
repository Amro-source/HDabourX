#include <zlib.h>
#include <iostream>
#include <vector>
#include <iomanip>
#include <sstream>

int main() {
    // Original data
    const char* input = "Hello, World!";
    size_t input_length = std::strlen(input);

    // Compress the data
    std::vector<Bytef> compressed_buffer(compressBound(input_length));
    uLongf compressed_size = compressed_buffer.size();

    int result = compress(
        compressed_buffer.data(),
        &compressed_size,
        reinterpret_cast<const Bytef*>(input),
        input_length
    );

    if (result != Z_OK) {
        std::cerr << "Compression failed: " << result << std::endl;
        return 1;
    }

    // Show compressed data (as hex)
    std::cout << "Compressed data (hex): ";
    for (size_t i = 0; i < compressed_size; ++i) {
        std::cout << std::hex << std::setw(2) << std::setfill('0')
            << static_cast<int>(compressed_buffer[i]);
    }
    std::cout << std::endl;

    // Decompress the data
    std::vector<Bytef> decompressed_buffer(input_length);
    uLongf decompressed_size = decompressed_buffer.size();

    result = uncompress(
        decompressed_buffer.data(),
        &decompressed_size,
        compressed_buffer.data(),
        compressed_size
    );

    if (result != Z_OK) {
        std::cerr << "Decompression failed: " << result << std::endl;
        return 1;
    }

    // Convert decompressed data back to string
    std::string decompressed_str(reinterpret_cast<char*>(decompressed_buffer.data()), decompressed_size);

    std::cout << "Original: " << input << std::endl;
    std::cout << "Decompressed: " << decompressed_str << std::endl;

    return 0;
}
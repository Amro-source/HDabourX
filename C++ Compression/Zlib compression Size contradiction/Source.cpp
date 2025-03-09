#include <zlib.h>
#include <iostream>
#include <vector>
#include <iomanip>
#include <sstream>

void print_hex(const std::string& label, const Bytef* data, size_t size) {
    std::cout << label << " (hex): ";
    for (size_t i = 0; i < size; ++i) {
        std::cout << std::hex << std::setw(2) << std::setfill('0')
            << static_cast<int>(data[i]);
    }
    std::cout << std::endl;
}

void print_string(const std::string& label, const Bytef* data, size_t size) {
    std::string str(reinterpret_cast<const char*>(data), size);
    std::cout << label << " (string): " << str << std::endl;
}

int main() {
    // Original data
    const char* input = "Hello, World!";
    std::string original(input);
    size_t input_length = original.size();

    // Compression
    std::vector<Bytef> compressed(compressBound(input_length));
    uLongf compressed_size = compressed.size();

    if (compress(compressed.data(), &compressed_size,
        reinterpret_cast<const Bytef*>(input), input_length) != Z_OK) {
        std::cerr << "Compression failed!" << std::endl;
        return 1;
    }

    // Show compressed data
    print_hex("Compressed", compressed.data(), compressed_size);
    print_string("Compressed", compressed.data(), compressed_size);

    // Decompression
    std::vector<Bytef> decompressed(input_length);
    uLongf decompressed_size = decompressed.size();

    if (uncompress(decompressed.data(), &decompressed_size,
        compressed.data(), compressed_size) != Z_OK) {
        std::cerr << "Decompression failed!" << std::endl;
        return 1;
    }

    // Show decompressed data
    print_hex("Decompressed", decompressed.data(), decompressed_size);
    print_string("Decompressed", decompressed.data(), decompressed_size);

    // Verify
    if (original == std::string(decompressed.begin(), decompressed.end())) {
        std::cout << "Verification: SUCCESS" << std::endl;
    }
    else {
        std::cout << "Verification: FAILED" << std::endl;
    }

    return 0;
}
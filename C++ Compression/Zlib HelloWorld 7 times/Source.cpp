#include <zlib.h>
#include <iostream>
#include <vector>
#include <iomanip>
#include <sstream>

void print_info(const std::string& label, const Bytef* data, size_t size) {
    std::cout << label << ":\n";
    std::cout << "  Size: " << size << " bytes\n";
    std::cout << "  Hex: ";
    for (size_t i = 0; i < size; ++i) {
        std::cout << std::hex << std::setw(2) << std::setfill('0')
            << static_cast<int>(data[i]);
    }
    std::cout << "\n  String: "
        << std::string(reinterpret_cast<const char*>(data), size)
        << std::endl << std::endl;
}

int main() {
    // Create larger, compressible data
    std::string original;
    for (int i = 0; i < 10; ++i) {
        original += "Hello, World! ";
    }
    original += "Hello, World!";
    size_t input_length = original.size();

    // Compression
    std::vector<Bytef> compressed(compressBound(input_length));
    uLongf compressed_size = compressed.size();

    if (compress(compressed.data(), &compressed_size,
        reinterpret_cast<const Bytef*>(original.data()), input_length) != Z_OK) {
        std::cerr << "Compression failed!" << std::endl;
        return 1;
    }

    // Show compressed data
    print_info("Original Data", reinterpret_cast<const Bytef*>(original.data()), input_length);
    print_info("Compressed Data", compressed.data(), compressed_size);

    // Decompression
    std::vector<Bytef> decompressed(input_length);
    uLongf decompressed_size = decompressed.size();

    if (uncompress(decompressed.data(), &decompressed_size,
        compressed.data(), compressed_size) != Z_OK) {
        std::cerr << "Decompression failed!" << std::endl;
        return 1;
    }

    // Verify
    bool verified = (original == std::string(decompressed.begin(), decompressed.end()));
    std::cout << "Verification: " << (verified ? "SUCCESS" : "FAILED") << std::endl;

    return 0;
}
#include <zlib.h>
#include <iostream>
#include <vector>

int main() {
    std::string data = "This is a test. ";
    data += data + data; // Create larger input

    z_stream strm;
    strm.zalloc = Z_NULL;
    strm.zfree = Z_NULL;
    strm.opaque = Z_NULL;

    deflateInit(&strm, Z_DEFAULT_COMPRESSION);

    std::vector<Bytef> output(1024);
    strm.avail_in = data.size();
    strm.next_in = (Bytef*)data.data();
    strm.avail_out = output.size();
    strm.next_out = output.data();

    // Stream in chunks
    int ret = deflate(&strm, Z_NO_FLUSH);
    if (ret != Z_OK) { /* handle error */ }

    // Finalize compression
    ret = deflate(&strm, Z_FINISH);
    deflateEnd(&strm);

    size_t compressed_size = output.size() - strm.avail_out;
    std::cout << "Streaming compressed size: " << compressed_size << " bytes" << std::endl;
    return 0;
}
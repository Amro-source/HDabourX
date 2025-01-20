#include <iostream>
#include <fstream>
#include <string>
#include <dirent.h>
#include <sys/types.h>
#include <cstring>

void appendLineToFile(const std::string& filePath, const std::string& line) {
    std::ofstream file(filePath, std::ios::app); // Open file in append mode
    if (file.is_open()) {
        file << line << std::endl; // Append the line
        file.close();
    }
    else {
        std::cerr << "Could not open file: " << filePath << std::endl;
    }
}

bool hasExtension(const std::string& filename, const std::string& extension) {
    return filename.size() >= extension.size() &&
        filename.compare(filename.size() - extension.size(), extension.size(), extension) == 0;
}

int main(int argc, char* argv[]) {
    if (argc != 4) {
        std::cerr << "Usage: " << argv[0] << " <path> <extension> <line_to_add>" << std::endl;
        return 1;
    }

    std::string path = argv[1];
    std::string extension = argv[2];
    std::string lineToAdd = argv[3];

    // Ensure the extension starts with a dot
    if (!extension.empty() && extension[0] != '.') {
        extension = '.' + extension;
    }

    DIR* dir = opendir(path.c_str());
    if (dir == nullptr) {
        std::cerr << "Could not open directory: " << path << std::endl;
        return 1;
    }

    struct dirent* entry;
    while ((entry = readdir(dir)) != nullptr) {
        if (entry->d_type == DT_REG) { // Check if it's a regular file
            std::string filename = entry->d_name;
            if (hasExtension(filename, extension)) {
                std::string fullPath = path + "/" + filename;
                appendLineToFile(fullPath, lineToAdd);
            }
        }
    }

    closedir(dir);
    return 0;
}
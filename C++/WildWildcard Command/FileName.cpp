#include <iostream>
#include <fstream>
#include <string>
#include <dirent.h>
#include <cstring>
#include <sys/types.h>

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

bool matchesWildcard(const std::string& filename, const std::string& pattern) {
    // Simple wildcard matching function
    // This function only supports '*' as a wildcard
    if (pattern == "*") return true;

    size_t starPos = pattern.find('*');
    if (starPos == std::string::npos) {
        return filename == pattern; // No wildcard, direct comparison
    }

    // Check the part before the '*'
    std::string prefix = pattern.substr(0, starPos);
    if (filename.find(prefix) != 0) {
        return false; // Prefix does not match
    }

    // Check the part after the '*'
    std::string suffix = pattern.substr(starPos + 1);
    if (suffix.empty()) {
        return true; // Only prefix, matches everything after
    }

    return filename.find(suffix) != std::string::npos; // Check if suffix exists
}

int main(int argc, char* argv[]) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <path\\*.ext> <line_to_add>" << std::endl;
        return 1;
    }

    std::string path = argv[1];
    std::string lineToAdd = argv[2];

    // Extract the directory and the wildcard pattern
    std::string directory = path.substr(0, path.find_last_of("\\/"));
    std::string pattern = path.substr(path.find_last_of("\\/") + 1);

    DIR* dir = opendir(directory.c_str());
    if (dir == nullptr) {
        std::cerr << "Could not open directory: " << directory << std::endl;
        return 1;
    }

    struct dirent* entry;
    while ((entry = readdir(dir)) != nullptr) {
        if (entry->d_type == DT_REG) { // Check if it's a regular file
            std::string filename = entry->d_name;
            if (matchesWildcard(filename, pattern)) {
                std::string fullPath = directory + "/" + filename;
                appendLineToFile(fullPath, lineToAdd);
                std::cout << "Appended to: " << fullPath << std::endl; // Debug output
            }
        }
    }

    closedir(dir);
    return 0;
}
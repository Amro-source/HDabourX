#include <iostream>
#include <fstream>
#include <string>
#include <unordered_map>
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
    if (pattern == "*") return true;

    size_t starPos = pattern.find('*');
    if (starPos == std::string::npos) {
        return filename == pattern; // No wildcard, direct comparison
    }

    std::string prefix = pattern.substr(0, starPos);
    if (filename.find(prefix) != 0) {
        return false; // Prefix does not match
    }

    std::string suffix = pattern.substr(starPos + 1);
    if (suffix.empty()) {
        return true; // Only prefix, matches everything after
    }

    return filename.find(suffix) != std::string::npos; // Check if suffix exists
}

void printUsage(const char* programName) {
    std::cout << "Usage: " << programName << " -i <path\\*.ext> -e <extension> -l <line_to_add>\n";
    std::cout << "Example: " << programName << " -i ./files\\*.txt -e txt -l \"New line\"\n";
}

int main(int argc, char* argv[]) {
    if (argc < 7) {
        printUsage(argv[0]);
        return 1;
    }

    std::unordered_map<std::string, std::string> args;
    for (int i = 1; i < argc; i += 2) {
        if (i + 1 < argc) {
            args[argv[i]] = argv[i + 1];
        }
        else {
            std::cerr << "Invalid argument: " << argv[i] << std::endl;
            printUsage(argv[0]);
            return 1;
        }
    }

    if (args.find("-i") == args.end() || args.find("-e") == args.end() || args.find("-l") == args.end()) {
        std::cerr << "Missing required arguments." << std::endl;
        printUsage(argv[0]);
        return 1;
    }

    std::string path = args["-i"];
    std::string extension = args["-e"];
    std::string lineToAdd = args["-l"];

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

            size_t extPos = filename.find_last_of('.');
            if (extPos != std::string::npos) {
                std::string fileExt = filename.substr(extPos + 1);
                if (fileExt == extension && matchesWildcard(filename, pattern)) {
                    std::string fullPath = directory + "/" + filename;
                    appendLineToFile(fullPath, lineToAdd);
                    std::cout << "Appended to: " << fullPath << std::endl;
                }
            }
        }
    }

    closedir(dir);
    return 0;
}

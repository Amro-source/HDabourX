#include <iostream>
#include <fstream>
#include <string>

int main(int argc, char* argv[]) {
    // Check if the correct number of arguments is provided
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <filename> <line_to_append>" << std::endl;
        return 1;
    }

    // Retrieve the filename and the line to append from command-line arguments
    std::string filename = argv[1];
    std::string lineToAppend = argv[2];

    // Open the file in append mode
    std::ofstream file(filename, std::ios::app);
    if (!file) {
        std::cerr << "Error: Could not open file " << filename << " for appending." << std::endl;
        return 1;
    }

    // Append the line followed by a newline character
    file << lineToAppend << '\n';

    // Close the file
    file.close();

    std::cout << "Line appended successfully to " << filename << std::endl;
    return 0;
}

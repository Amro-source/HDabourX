#include <iostream>
#include <string>
#include <cstring>
#include <cstdlib>
#include <winsock2.h>  // For Windows
#include <opencv2/opencv.hpp>  // For OpenCV image processing
#include <filesystem>  // For modern path handling

#pragma comment(lib, "Ws2_32.lib")  // Windows-specific

using namespace cv;
namespace fs = std::filesystem;

// Function to initialize the server socket
SOCKET start_server(int port) {
    WSADATA wsaData;
    WSAStartup(MAKEWORD(2, 2), &wsaData);

    SOCKET server_socket = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    if (server_socket == INVALID_SOCKET) {
        std::cerr << "Error creating socket!" << std::endl;
        WSACleanup();
        return INVALID_SOCKET;
    }

    sockaddr_in server_address;
    server_address.sin_family = AF_INET;
    server_address.sin_port = htons(port);  // Bind to the given port
    server_address.sin_addr.s_addr = INADDR_ANY;

    if (bind(server_socket, (sockaddr*)&server_address, sizeof(server_address)) == SOCKET_ERROR) {
        std::cerr << "Error binding socket!" << std::endl;
        closesocket(server_socket);
        WSACleanup();
        return INVALID_SOCKET;
    }

    return server_socket;
}

// Function to listen for incoming client connections
SOCKET listen_for_clients(SOCKET server_socket) {
    if (listen(server_socket, 1) == SOCKET_ERROR) {
        std::cerr << "Error listening on socket!" << std::endl;
        closesocket(server_socket);
        WSACleanup();
        return INVALID_SOCKET;
    }

    std::cout << "Server is listening for clients..." << std::endl;

    sockaddr_in client_address;
    int client_address_size = sizeof(client_address);
    SOCKET client_socket = accept(server_socket, (sockaddr*)&client_address, &client_address_size);

    if (client_socket == INVALID_SOCKET) {
        std::cerr << "Error accepting client connection!" << std::endl;
        closesocket(server_socket);
        WSACleanup();
        return INVALID_SOCKET;
    }

    return client_socket;
}

// Function to receive a message from the client
std::string receive_message(SOCKET client_socket) {
    char buffer[1024];
    int bytes_received = recv(client_socket, buffer, sizeof(buffer) - 1, 0);
    if (bytes_received > 0) {
        buffer[bytes_received] = '\0';  // Null-terminate the received data
        return std::string(buffer);
    }
    return "";
}

// Function to send a message to the client
void send_message(SOCKET client_socket, const std::string& message) {
    send(client_socket, message.c_str(), message.length(), 0);
}

// Function to apply a blue filter to an image
Mat apply_blue_filter(const Mat& image) {
    Mat filtered_image;
    std::vector<Mat> channels(3);
    split(image, channels);

    // Set green and red channels to zero
    channels[1] = Mat::zeros(image.rows, image.cols, CV_8UC1);
    channels[2] = Mat::zeros(image.rows, image.cols, CV_8UC1);

    merge(channels, filtered_image);
    return filtered_image;
}

// Generate processed file name
std::string get_processed_filename(const std::string& input_path) {
    fs::path original_path(input_path);
    std::string processed_name = "processed_" + original_path.filename().string();
    fs::path output_path = fs::current_path() / processed_name;  // Save in current directory
    return output_path.string();
}

// Main function to run the server
int main() {
    int port = 12345;  // Updated to match the Python script
    SOCKET server_socket = start_server(port);
    if (server_socket == INVALID_SOCKET) return -1;

    SOCKET client_socket = listen_for_clients(server_socket);
    if (client_socket == INVALID_SOCKET) return -1;

    std::cout << "Waiting for image name from client..." << std::endl;

    // Receive the image name from the client
    std::string image_name = receive_message(client_socket);
    if (image_name.empty()) {
        std::cerr << "No image name received!" << std::endl;
        closesocket(client_socket);
        closesocket(server_socket);
        WSACleanup();
        return -1;
    }

    std::cout << "Received image name: " << image_name << std::endl;

    // Read the input image
    Mat image = imread(image_name, IMREAD_COLOR);
    if (image.empty()) {
        std::cerr << "Error reading input image: " << image_name << std::endl;
        send_message(client_socket, "Error: Could not read image.");
        closesocket(client_socket);
        closesocket(server_socket);
        WSACleanup();
        return -1;
    }

    // Apply blue filter
    Mat blue_image = apply_blue_filter(image);

    // Save the processed image
    std::string output_name = get_processed_filename(image_name);
    imwrite(output_name, blue_image);

    // Notify the client
    send_message(client_socket, "Image processed and saved as " + output_name);

    closesocket(client_socket);
    closesocket(server_socket);
    WSACleanup();

    return 0;
}

#include <iostream>
#include <string>
#include <cstring>
#include <cstdlib>
#include <winsock2.h>  // For Windows
#include <opencv2/opencv.hpp>  // For OpenCV image processing
#include <filesystem>  // For modern file path handling

#pragma comment(lib, "Ws2_32.lib")  // Link Winsock library

using namespace cv;
namespace fs = std::filesystem;

// Function to initialize the server socket
SOCKET start_server(int port) {
    WSADATA wsaData;
    if (WSAStartup(MAKEWORD(2, 2), &wsaData) != 0) {
        std::cerr << "Error initializing Winsock! Error Code: " << WSAGetLastError() << std::endl;
        return INVALID_SOCKET;
    }

    SOCKET server_socket = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    if (server_socket == INVALID_SOCKET) {
        std::cerr << "Error creating socket! Error Code: " << WSAGetLastError() << std::endl;
        WSACleanup();
        return INVALID_SOCKET;
    }

    sockaddr_in server_address;
    server_address.sin_family = AF_INET;
    server_address.sin_port = htons(port);
    server_address.sin_addr.s_addr = INADDR_ANY;

    if (bind(server_socket, (sockaddr*)&server_address, sizeof(server_address)) == SOCKET_ERROR) {
        std::cerr << "Error binding socket! Error Code: " << WSAGetLastError() << std::endl;
        closesocket(server_socket);
        WSACleanup();
        return INVALID_SOCKET;
    }

    if (listen(server_socket, 1) == SOCKET_ERROR) {
        std::cerr << "Error listening on socket! Error Code: " << WSAGetLastError() << std::endl;
        closesocket(server_socket);
        WSACleanup();
        return INVALID_SOCKET;
    }

    std::cout << "Server started and listening on port: " << port << std::endl;
    return server_socket;
}

// Function to accept a client connection
SOCKET accept_client(SOCKET server_socket) {
    sockaddr_in client_address;
    int client_address_size = sizeof(client_address);
    SOCKET client_socket = accept(server_socket, (sockaddr*)&client_address, &client_address_size);

    if (client_socket == INVALID_SOCKET) {
        std::cerr << "Error accepting client connection! Error Code: " << WSAGetLastError() << std::endl;
    }
    else {
        std::cout << "Client connected!" << std::endl;
    }

    return client_socket;
}

// Function to receive a message from the client
std::string receive_message(SOCKET client_socket) {
    char buffer[1024];
    int bytes_received = recv(client_socket, buffer, sizeof(buffer) - 1, 0);
    if (bytes_received > 0) {
        buffer[bytes_received] = '\0';  // Null-terminate the received data
        std::cout << "Received image path: " << buffer << std::endl;
        return std::string(buffer);
    }
    return "";
}

// Function to send a message to the client
void send_message(SOCKET client_socket, const std::string& message) {
    send(client_socket, message.c_str(), message.length(), 0);
    std::cout << "Sent message to client: " << message << std::endl;
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
    fs::path output_path = original_path.parent_path().parent_path() / "STAGE_3" / processed_name;
    fs::create_directories(output_path.parent_path());  // Ensure STAGE_3 folder exists
    return output_path.string();
}

// Main function to run the server
int main() {
    int port = 12345;

    SOCKET server_socket = start_server(port);
    if (server_socket == INVALID_SOCKET) return -1;

    while (true) {
        SOCKET client_socket = accept_client(server_socket);
        if (client_socket == INVALID_SOCKET) continue;

        std::string image_path = receive_message(client_socket);
        if (image_path.empty()) {
            send_message(client_socket, "Error: No image path received.");
            closesocket(client_socket);
            continue;
        }

        std::cout << "Processing image: " << image_path << std::endl;

        Mat image = imread(image_path, IMREAD_COLOR);
        if (image.empty()) {
            std::cerr << "Error: Failed to read image: " << image_path << std::endl;
            send_message(client_socket, "Error: Failed to read image.");
            closesocket(client_socket);
            continue;
        }

        Mat blue_image = apply_blue_filter(image);
        std::string output_path = get_processed_filename(image_path);

        if (!imwrite(output_path, blue_image)) {
            std::cerr << "Error: Failed to save processed image to: " << output_path << std::endl;
            send_message(client_socket, "Error: Failed to save processed image.");
        }
        else {
            std::cout << "Image processed and saved to: " << output_path << std::endl;
            send_message(client_socket, "Processed image saved as: " + output_path);
        }

        closesocket(client_socket);
        std::cout << "Client disconnected." << std::endl;
    }

    closesocket(server_socket);
    WSACleanup();
    return 0;
}

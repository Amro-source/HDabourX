#include <iostream>
#include <string>
#include <cstring>
#include <cstdlib>
#include <winsock2.h>  // For Windows
// For Linux or macOS, replace with <sys/socket.h> and <netinet/in.h>

#pragma comment(lib, "Ws2_32.lib")  // Windows-specific

int main() {
    WSADATA wsaData;
    WSAStartup(MAKEWORD(2, 2), &wsaData);

    SOCKET server_socket = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    if (server_socket == INVALID_SOCKET) {
        std::cerr << "Error creating socket!" << std::endl;
        WSACleanup();
        return -1;
    }

    sockaddr_in server_address;
    server_address.sin_family = AF_INET;
    server_address.sin_port = htons(65432);  // Same port as Python client
    server_address.sin_addr.s_addr = INADDR_ANY;

    if (bind(server_socket, (sockaddr*)&server_address, sizeof(server_address)) == SOCKET_ERROR) {
        std::cerr << "Error binding socket!" << std::endl;
        closesocket(server_socket);
        WSACleanup();
        return -1;
    }

    if (listen(server_socket, 1) == SOCKET_ERROR) {
        std::cerr << "Error listening on socket!" << std::endl;
        closesocket(server_socket);
        WSACleanup();
        return -1;
    }

    std::cout << "Server is listening on port 65432..." << std::endl;

    sockaddr_in client_address;
    int client_address_size = sizeof(client_address);
    SOCKET client_socket = accept(server_socket, (sockaddr*)&client_address, &client_address_size);

    if (client_socket == INVALID_SOCKET) {
        std::cerr << "Error accepting client connection!" << std::endl;
        closesocket(server_socket);
        WSACleanup();
        return -1;
    }

    std::cout << "Client connected!" << std::endl;

    char buffer[1024];
    int bytes_received = recv(client_socket, buffer, sizeof(buffer), 0);
    if (bytes_received > 0) {
        buffer[bytes_received] = '\0';  // Null-terminate the received data
        std::cout << "Received from client: " << buffer << std::endl;

        // Respond to the client
        std::string response = "Hello from C++!";
        send(client_socket, response.c_str(), response.length(), 0);
    }

    closesocket(client_socket);
    closesocket(server_socket);
    WSACleanup();

    return 0;
}

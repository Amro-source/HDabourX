#include <iostream>
#include <string>
#include <cstring>
#include <cstdlib>
#include <winsock2.h>  // For Windows
// For Linux or macOS, replace with <sys/socket.h> and <netinet/in.h>

#pragma comment(lib, "Ws2_32.lib")  // Windows-specific

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
    int bytes_received = recv(client_socket, buffer, sizeof(buffer), 0);
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

// Main function to run the server
int main() {
    int port = 65432;
    SOCKET server_socket = start_server(port);
    if (server_socket == INVALID_SOCKET) return -1;

    SOCKET client_socket = listen_for_clients(server_socket);
    if (client_socket == INVALID_SOCKET) return -1;

    std::string received_message = receive_message(client_socket);
    std::cout << "Received from client: " << received_message << std::endl;

    send_message(client_socket, "Hello from C++!");

    closesocket(client_socket);
    closesocket(server_socket);
    WSACleanup();

    return 0;
}

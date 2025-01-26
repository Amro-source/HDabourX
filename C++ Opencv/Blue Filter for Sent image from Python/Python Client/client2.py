import socket
import os

def send_image_request(image_name):
    HOST = '127.0.0.1'  # Localhost
    PORT = 12345         # Port used by the C++ application

    # Get the absolute path of the image
    absolute_path = os.path.abspath(image_name)

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((HOST, PORT))
        s.sendall(absolute_path.encode())
        data = s.recv(1024)
        print(f"Received: {data.decode()}")

if __name__ == "__main__":
    # Replace with the image file name
    image_name = "input.jpeg"
    send_image_request(image_name)

import socket

STAGE_1 = "STAGE_1"
STAGE_3 = "STAGE_3"

def send_image_request(image_name):
    HOST = '127.0.0.1'  # Localhost
    PORT = 12345         # Port used by the C++ application

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((HOST, PORT))
        s.sendall(image_name.encode())
        data = s.recv(1024)

    print(f"Received: {data.decode()}")

if __name__ == "__main__":
    # Example: Request processing of a specific image
    image_name = "example.jpg"  # Replace with actual image file name in STAGE_1
    send_image_request(image_name)

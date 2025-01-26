import socket
import os

def send_image_request(image_path):
    HOST = '127.0.0.1'  # Localhost
    PORT = 12345         # Port used by the C++ application

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((HOST, PORT))
        s.sendall(image_path.encode())

        # Receive server response
        data = s.recv(1024)
        print(f"Received from server: {data.decode()}")

def main():
    # Define the STAGE_1 and STAGE_3 folders
    stage_1_folder = os.path.join(os.getcwd(), "STAGE_1")
    stage_3_folder = os.path.join(os.getcwd(), "STAGE_3")
    os.makedirs(stage_3_folder, exist_ok=True)  # Ensure STAGE_3 exists

    # Iterate through all image files in STAGE_1
    for file_name in os.listdir(stage_1_folder):
        file_path = os.path.join(stage_1_folder, file_name)
        if os.path.isfile(file_path) and file_name.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff')):
            print(f"Sending image: {file_path}")
            send_image_request(file_path)

if __name__ == "__main__":
    main()

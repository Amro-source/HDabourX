import subprocess
import os

def main():
    # Define paths
    input_folder = os.path.join(os.getcwd(), "STAGE_1")
    output_folder = os.path.join(os.getcwd(), "STAGE_3")
    executable = os.path.join(os.getcwd(), "opencv_processor.exe")  # Replace with the actual path to your exe

    # Ensure input and output folders exist
    if not os.path.exists(input_folder):
        print(f"Error: Input folder does not exist: {input_folder}")
        return
    os.makedirs(output_folder, exist_ok=True)

    # Run the C++ executable
    print("Starting image processing...")
    try:
        subprocess.run([executable, input_folder, output_folder], check=True)
        print("Image processing completed successfully from Opencv C++ App!")
        print("Execution is finished")
    except subprocess.CalledProcessError as e:
        print(f"Error during image processing: {e}")
    except FileNotFoundError:
        print(f"Error: Executable not found: {executable}")

if __name__ == "__main__":
    main()


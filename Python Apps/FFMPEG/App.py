import ffmpeg

# Example: Convert a video file to a different format
input_file = "input.mp4"
output_file = "output.avi"

try:
    (
        ffmpeg
        .input(input_file)
        .output(output_file)
        .run()
    )
    print("Conversion completed successfully!")
except ffmpeg.Error as e:
    print(f"An error occurred: {e}")

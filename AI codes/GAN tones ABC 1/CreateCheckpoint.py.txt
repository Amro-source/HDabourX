import os
import time

# Define the checkpoint directory
CHECKPOINT_DIR = os.path.abspath("checkpoints")

def ensure_directory_exists(directory):
    """
    Ensures that the specified directory exists.
    Retries up to 5 times if creation fails.
    """
    retries = 5
    while retries > 0:
        try:
            # Attempt to create the directory
            os.makedirs(directory, exist_ok=True)
            print(f"Successfully created or verified directory: {directory}")
            return True
        except FileExistsError:
            print(f"Directory already exists: {directory}")
            return True
        except Exception as e:
            print(f"Failed to create directory {directory}: {e}. Retrying...")
            retries -= 1
            time.sleep(1)
    raise RuntimeError(f"Could not create directory {directory} after multiple attempts.")

if __name__ == "__main__":
    print(f"Attempting to create checkpoint directory at: {CHECKPOINT_DIR}")

    try:
        # Ensure the directory exists
        ensure_directory_exists(CHECKPOINT_DIR)

        # Verify the directory exists
        if os.path.exists(CHECKPOINT_DIR):
            print(f"Checkpoint directory '{CHECKPOINT_DIR}' successfully created or already exists.")
        else:
            print(f"Checkpoint directory '{CHECKPOINT_DIR}' still does not exist after creation attempt.")
    except Exception as e:
        print(f"An error occurred: {e}")
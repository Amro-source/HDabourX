import time
import threading
import sys

class Stopwatch:
    def __init__(self):
        self.start_time = None
        self.elapsed_time = 0
        self.running = False
        self.lock = threading.Lock()

    def start(self):
        if not self.running:
            self.start_time = time.time() - self.elapsed_time
            self.running = True
            threading.Thread(target=self._update_display, daemon=True).start()

    def pause(self):
        if self.running:
            self.running = False
            self.elapsed_time = time.time() - self.start_time

    def reset(self):
        self.running = False
        self.elapsed_time = 0

    def _update_display(self):
        while self.running:
            with self.lock:
                current_elapsed = time.time() - self.start_time
                minutes = int(current_elapsed // 60)
                seconds = int(current_elapsed % 60)
                sys.stdout.write(f"\rTime: {minutes:02d}:{seconds:02d} | Elapsed: {current_elapsed:.1f} sec")
                sys.stdout.flush()
            time.sleep(0.1)

def main():
    stopwatch = Stopwatch()
    
    print("Enhanced Stopwatch with Elapsed Time")
    print("Commands: start, pause, reset, exit")
    print("\rTime: 00:00 | Elapsed: 0.0 sec", end="")
    
    while True:
        command = input("\n> ").lower()
        
        if command == "start":
            stopwatch.start()
        elif command == "pause":
            stopwatch.pause()
        elif command == "reset":
            stopwatch.reset()
            sys.stdout.write("\rTime: 00:00 | Elapsed: 0.0 sec")
            sys.stdout.flush()
        elif command == "exit":
            break
        else:
            print("Invalid command. Please use: start, pause, reset, exit")

if __name__ == "__main__":
    main()
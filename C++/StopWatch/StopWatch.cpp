#include <iostream>
#include <chrono>
#include <thread>
#include <atomic>
#include <iomanip>
#include <mutex>
#include <string>

class Stopwatch {
private:
    std::chrono::time_point<std::chrono::steady_clock> start_time;
    std::chrono::duration<double> elapsed_time{ 0 };
    std::atomic<bool> running{ false };
    std::mutex mtx;

public:
    void start() {
        if (!running) {
            start_time = std::chrono::steady_clock::now() - std::chrono::duration_cast<std::chrono::steady_clock::duration>(elapsed_time);
            running = true;
            std::thread([this]() { this->update_display(); }).detach();
        }
    }

    void pause() {
        if (running) {
            running = false;
            std::lock_guard<std::mutex> lock(mtx);
            elapsed_time = std::chrono::steady_clock::now() - start_time;
        }
    }

    void reset() {
        running = false;
        elapsed_time = std::chrono::duration<double>(0);
        std::cout << "\rTime: 00:00 | Elapsed: 0.0 sec" << std::flush;
    }

    void update_display() {
        while (running) {
            std::lock_guard<std::mutex> lock(mtx);
            auto current_elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - start_time).count();

            int minutes = static_cast<int>(current_elapsed) / 60;
            int seconds = static_cast<int>(current_elapsed) % 60;

            std::cout << "\rTime: "
                << std::setw(2) << std::setfill('0') << minutes << ":"
                << std::setw(2) << std::setfill('0') << seconds
                << " | Elapsed: " << std::fixed << std::setprecision(1)
                << current_elapsed << " sec" << std::flush;

            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
    }
};

void display_help() {
    std::cout << "\nStopwatch Commands:\n"
        << "start - Begin timing\n"
        << "pause - Pause the stopwatch\n"
        << "reset - Reset to zero\n"
        << "exit  - Quit the program\n";
}

int main() {
    Stopwatch stopwatch;
    std::string command;

    std::cout << "C++ Console Stopwatch\n";
    display_help();
    std::cout << "\rTime: 00:00 | Elapsed: 0.0 sec" << std::flush;

    while (true) {
        std::cout << "\n> ";
        std::getline(std::cin, command);

        if (command == "start") {
            stopwatch.start();
        }
        else if (command == "pause") {
            stopwatch.pause();
        }
        else if (command == "reset") {
            stopwatch.reset();
        }
        else if (command == "exit") {
            break;
        }
        else if (command == "help") {
            display_help();
        }
        else {
            std::cout << "Invalid command. Type 'help' for available commands.\n";
        }
    }

    return 0;
}
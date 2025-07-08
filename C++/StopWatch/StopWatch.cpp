


#define _CRT_SECURE_NO_WARNINGS

#include <iostream>
#include <chrono>
#include <thread>
#include <atomic>
#include <iomanip>
#include <mutex>
#include <string>
#include <format>
#include <assert.h>



tm timeTo_tm(std::chrono::system_clock::time_point time1)
{
    auto tt1 = std::chrono::system_clock::to_time_t(time1);
    auto p_tm1 = std::gmtime(&tt1);
    assert(nullptr != p_tm1);
    return *p_tm1;
}


class Stopwatch {
private:
    std::chrono::time_point<std::chrono::system_clock> start_time;
    std::chrono::duration<double> elapsed_time{ 0 };
    std::atomic<bool> running{ false };
    std::mutex mtx;

public:

    std::string name;

    void start() {
        if (!running) {
            start_time = std::chrono::system_clock::now() - std::chrono::duration_cast<std::chrono::system_clock::duration>(elapsed_time);
            running = true;
            display_header("Started");
            std::thread([this]() { this->update_display(); }).detach();
        }
    }

    void pause() {
        if (running) {
            running = false;
            std::lock_guard<std::mutex> lock(mtx);
            elapsed_time = std::chrono::system_clock::now() - start_time;
        }
        display_header("Paused");
    }

    void reset() {
        if (running) {
            running = false;
            std::lock_guard<std::mutex> lock(mtx);
            elapsed_time = std::chrono::system_clock::now() - start_time;
        }

        elapsed_time = std::chrono::duration<double>(0);
        display_header("Reset");

        if (!running) {
            start_time = std::chrono::system_clock::now() - std::chrono::duration_cast<std::chrono::system_clock::duration>(elapsed_time);
            running = true;
            std::thread([this]() { this->update_display(); }).detach();
        }
    }

    void display_header(const std::string str1)
    {
        auto tm1 = timeTo_tm(std::chrono::system_clock::now());
        std::cout << "\n" << name << " " << str1 << " at Time: "
            << std::setw(2) << std::setfill('0') << tm1.tm_hour + 3 << ":"
            << std::setw(2) << std::setfill('0') << tm1.tm_min << ":"
            << std::setw(2) << std::setfill('0') << tm1.tm_sec << "\n";
    }


    void update_display() {
        while (running) {
            std::lock_guard<std::mutex> lock(mtx);
            auto current_elapsed = std::chrono::duration<double>(std::chrono::system_clock::now() - start_time).count();

            int hours = static_cast<int>(current_elapsed) / (60 * 60);
            
            float total_minutes_f = static_cast<float>(current_elapsed) / 60;
            int total_minutes = static_cast<int>(total_minutes_f);

            int minutes = total_minutes % 60;
            int seconds = static_cast<int>(current_elapsed) % 60;

            std::cout << "\r\t  Elapsed: "
                << std::setw(2) << std::setfill('0') << hours << ":"
                << std::setw(2) << std::setfill('0') << minutes << ":"
                << std::setw(2) << std::setfill('0') << seconds

                << " | Total Minutes: " << std::fixed << std::setprecision(2)
                << total_minutes_f << " mins " << "                          " << std::flush;

            //std::this_thread::sleep_for(std::chrono::milliseconds(100));
            std::this_thread::sleep_for(std::chrono::milliseconds(500));
        }
    }
};

void display_help() {
    std::cout << "\nStopwatch Commands:\n"
        << "(s)tart - Begin timing\n"
        << "(p)ause - Pause the stop_watch\n"
        << "rese(t) - Reset to zero\n"
        << "e(x)it  - Quit the program\n";
}

int main() {


    Stopwatch stop_watch;
    std::string command;

    std::cout << "Console Stopwatch\n";

    std::cout << "\n\nEnter Stopwatch name:\n";
    std::getline(std::cin, stop_watch.name);

    display_help();


    while (true) {
        std::cout << "\n> ";
        std::getline(std::cin, command);

        if (command == "s") {
            stop_watch.start();
        }
        else if (command == "p") {
            stop_watch.pause();
        }
        else if (command == "t") {
            stop_watch.reset();
            //stop_watch.start();
        }
        else if (command == "x") {
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




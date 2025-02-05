//Here's an example C++ code that demonstrates two threads:
//
//1. Worker Thread : Performs a routine task(simulated by a loop that increments a counter).
//2. Logger Thread : Waits for a trigger signal and then writes a log message.


#include <iostream>
#include <thread>
#include <mutex>
#include <condition_variable>

std::mutex mtx;
std::condition_variable cv;
bool trigger = false;

void worker() {
    int counter = 0;
    while (true) {
        {
            std::lock_guard<std::mutex> lock(mtx);
            counter++;
            std::cout << "Worker: Counter = " << counter << std::endl;
        }
        std::this_thread::sleep_for(std::chrono::seconds(1));
        if (counter >= 5) {
            {
                std::lock_guard<std::mutex> lock(mtx);
                trigger = true;
            }
            cv.notify_one();
            break;
        }
    }
}

void logger() {
    std::unique_lock<std::mutex> lock(mtx);
    cv.wait(lock, [] { return trigger; });
    std::cout << "Logger: Writing log..." << std::endl;
    // Simulate log writing
    std::this_thread::sleep_for(std::chrono::seconds(2));
    std::cout << "Logger: Log written." << std::endl;
}

int main() {
    std::thread workerThread(worker);
    std::thread loggerThread(logger);

    workerThread.join();
    loggerThread.join();

    return 0;
}

//
//Explanation:
//
//-The worker function simulates a routine task by incrementing a counter every second.
//- When the counter reaches 5, the worker sets the trigger flag to true and notifies the logger thread using cv.notify_one().
//- The logger function waits for the trigger flag to become true using cv.wait(lock, [] { return trigger; }).Once triggered, it simulates writing a log message.
//- In the main function, we create two threads for the worker and logger functions and wait for them to finish using join().
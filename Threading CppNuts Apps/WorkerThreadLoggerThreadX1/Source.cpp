//Here's a simplified version of the code that uses two threads, one for the worker and one for the logger. 
//The logger thread waits for a trigger signal from the worker thread before writing a log message.


#include <iostream>
#include <thread>
#include <atomic>

std::atomic<bool> trigger(false);

void worker() {
    for (int i = 0; i < 5; i++) {
        std::cout << "Worker: " << i << std::endl;
    }
    trigger = true;
}

void logger() {
    while (!trigger) {
        // busy-waiting
    }
    std::cout << "Logger: Writing log..." << std::endl;
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
//-The worker function simulates some work by printing numbers from 0 to 4.
//- After finishing its work, the worker sets the trigger flag to true.
//- The logger function busy - waits until the trigger flag becomes true, then prints a log message.
//- In the main function, we create two threads for the worker and logger functions and wait for them to finish using join().
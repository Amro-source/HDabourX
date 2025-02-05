//No, you don't necessarily need to use a std::mutex with std::condition_variable. However, you 
//do need to ensure that the condition variable is used in a thread-safe manner.
//
//One way to do this without using a std::mutex is to use a std::atomic variable to protect 
//the condition variable.Here's an example:
//

#include <iostream>
#include <thread>
#include <condition_variable>
#include <atomic>
#include <mutex>

std::condition_variable cv;
std::atomic<bool> trigger(false);
std::mutex mtx;

void worker() {
    for (int i = 0; i < 5; i++) {
        std::cout << "Worker: " << i << std::endl;
    }
    trigger = true;
    cv.notify_one();
}

void logger() {
    std::unique_lock<std::mutex> lock(mtx);
    while (!trigger.load()) {
        cv.wait(lock);
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






//In this example, we use an std::atomic<bool> to protect the trigger variable.The logger function 
//uses a lambda function to wait for the trigger variable to become true, and then prints a log message.
//
//Note that this approach requires C++11 or later, as it uses std::atomic and lambda functions.
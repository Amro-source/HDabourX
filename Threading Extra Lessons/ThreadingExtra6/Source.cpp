//Here is the complete and corrected code :
//

#include <iostream>
#include <thread>
#include <mutex>
#include <chrono>

std::mutex mtx;

void shared_print(std::string msg, int id) {
    mtx.lock();
    std::cout << "Thread " << id << ": " << msg << std::endl;
    mtx.unlock();
}

void worker(int id) {
    shared_print("Starting work", id);
    // Simulate work
    std::this_thread::sleep_for(std::chrono::seconds(2));
    shared_print("Finished work", id);
}

int main() {
    // Create threads
    std::thread workerThreads[5];
    for (int i = 0; i < 5; i++) {
        workerThreads[i] = std::thread(worker, i);
    }

    // Join threads
    for (int i = 0; i < 5; i++) {
        workerThreads[i].join();
    }

    return 0;
}

//
//This code creates 5 worker threads that simulate work by sleeping for 2 seconds.
//The shared_print function is used to print messages to the console in a thread - safe manner
//using a mutex.The main thread waits for all worker threads to finish using the join method.
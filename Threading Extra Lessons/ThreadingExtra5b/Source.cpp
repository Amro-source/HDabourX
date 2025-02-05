//Here is the code from Lesson 5 of Corey Schafer's Threading Tutorial in C++:
//

#include <iostream>
#include <thread>
#include <mutex>

std::mutex mtx;

void print_numbers() {
    for (int i = 0; i < 10; i++) {
        mtx.lock();
        std::cout << i << std::endl;
        mtx.unlock();
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }
}

void print_letters() {
    for (char letter = 'a'; letter <= 'j'; letter++) {
        mtx.lock();
        std::cout << letter << std::endl;
        mtx.unlock();
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }
}

int main() {
    // Create threads
    std::thread thread1(print_numbers);
    std::thread thread2(print_letters);

    // Join threads
    thread1.join();
    thread2.join();

    return 0;
}
//
///*
//This code demonstrates the following concepts :
//
//-Creating multiple threads using the std::thread class
//- Using a mutex(std::mutex) to protect shared resources
//- Locking and unlocking the mutex using the lock() and unlock() methods
//
//Something new :
//
//    In this lesson, we learn about the concept of a "mutex" (short for "mutual exclusion").A mutex is a synchronization primitive that allows only one thread to access a shared resource at a time.
//
//    By using a mutex, we can prevent multiple threads from accessing the same resource simultaneously, which can lead to data corruption or other concurrency - related issues.
//
//    //In this example, we use a mutex to protect the std::cout statement, which is a shared resource that can be accessed by multiple threads.By locking the mutex before accessing the shared resource, we ensure that only one thre*/ad can access it at a time.
#include <iostream>
#include <thread>
#include <chrono>

void print_numbers() {
    for (int i = 0; i < 10; i++) {
        std::cout << i << std::endl;
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }
}

int main() {
    // Create a thread
    std::thread thread(print_numbers);

    // Check if the thread is joinable
    if (thread.joinable()) {
        std::cout << "Thread is joinable" << std::endl;
    }
    else {
        std::cout << "Thread is not joinable" << std::endl;
    }

    // Join the thread
    thread.join();

    // Check if the thread is joinable after joining
    if (thread.joinable()) {
        std::cout << "Thread is joinable after joining" << std::endl;
    }
    else {
        std::cout << "Thread is not joinable after joining" << std::endl;
    }

    return 0;
}


//This code demonstrates the following concepts :
//
//-Creating a thread using the std::thread class
//- Checking if a thread is joinable using the joinable() method
//- Joining a thread using the join() method
//
//The output of this code will be :
//
//
//Thread is joinable
//0
//1
//2
//3
//4
//5
//6
//7
//8
//9
//Thread is not joinable after joining
//
//
//Note that after joining the thread, it is no longer joinable.
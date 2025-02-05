//Here is the code from Lesson 4 of Corey Schafer's Threading Tutorial in C++:


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
    std::thread numberPrinterThread(print_numbers);

    // Check if the thread is joinable
    if (numberPrinterThread.joinable()) {
        std::cout << "Thread is joinable" << std::endl;
    }

    // Detach the thread
    // numberPrinterThread.detach();

    // Try to detach the thread again
    // numberPrinterThread.detach();

    // Join the thread
    numberPrinterThread.join();

    // Try to join the thread again
    // numberPrinterThread.join();

    return 0;
}

//
//This code demonstrates the following concepts :
//
//-Creating a thread using the std::thread class
//- Checking if a thread is joinable using the joinable() method
//- Detaching a thread using the detach() method
//- Joining a thread using the join() method
//
//Note that :
//
//-You can only detach or join a thread once.Trying to detach or join a thread 
//again will result in a runtime error.
//- If you detach a thread, you cannot join it later.If you join a thread, you cannot detach it later.
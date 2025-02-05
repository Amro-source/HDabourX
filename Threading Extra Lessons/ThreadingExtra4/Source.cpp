//Here is the code from Lesson 3 of Corey Schafer's Threading Tutorial in C++:
//

#include <iostream>
#include <thread>
#include <chrono>

void print_numbers() {
    for (int i = 0; i < 10; i++) {
        std::cout << i << std::endl;
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }
}

//std::thread numberPrinterThread(print_numbers);
//
//does execute the thread.
//
//When you create a std::thread object, the thread is automatically started, and the function passed to the constructor(in this case, print_numbers) is executed in a separate thread.
//
//This is known as "implicit start" or "automatic start", and it's a feature of the std::thread class in C++.
//
//So, as soon as the std::thread object is created, the thread is started, and the print_numbers function begins executing in the background.
//
//Note that if you want to delay the start of the thread, you can use the std::thread constructor with a std::launch parameter, like this:
//
//std::thread numberPrinterThread(std::launch::deferred, print_numbers);
//
//This will create the thread, but not start it until you explicitly call the join() or detach() method on the thread object.


int main() {
    // Create a thread
    std::thread numberPrinterThread(print_numbers);

    // Detach the thread
    numberPrinterThread.detach();

    // Check if the thread is joinable after detaching
    if (numberPrinterThread.joinable()) {
        std::cout << "Thread is joinable after detaching" << std::endl;
    }
    else {
        std::cout << "Thread is not joinable after detaching" << std::endl;
    }

    // Wait for the detached thread to finish
    std::this_thread::sleep_for(std::chrono::seconds(10));

    return 0;
}

//
//This code demonstrates the following concepts :
//
//-Creating a thread using the std::thread class
//- Detaching a thread using the detach() method
//- Checking if a thread is joinable using the joinable() method
//
//Note that when a thread is detached, it is no longer joinable, and the join() method cannot be called on it.Also, the main thread will not wait for the detached thread to finish, so we use std::this_thread::sleep_for() to wait for the detached thread to finish.

//thread.join() does indeed wait for the thread to finish execution, but it doesn't start the thread.
//
//When you create a std::thread object, the thread is automatically started, and the function passed to the constructor begins executing in a separate thread.
//
//thread.join() is used to wait for the thread to finish execution, and it blocks the calling thread until the target thread has completed.
//
//Here's an example to illustrate the difference:
//
//std::thread thread(print_numbers);
//
//// At this point, the thread has already started executing print_numbers
//
//thread.join(); // Wait for the thread to finish execution
//
//So, to summarize :
//
//-Creating a std::thread object starts the thread automatically.
//- thread.join() waits for the thread to finish execution, but doesn't start the thread.
//


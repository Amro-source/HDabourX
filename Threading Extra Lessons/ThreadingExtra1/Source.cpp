#include <iostream>
#include <thread>
#include <chrono>
//
///*The code is an example of multithreading in C++.Multithreading is a way to run multiple tasks concurrently, improving the performance and responsiveness of a program.
//
//Here's a breakdown of the code:
//
//1. We define two functions : print_numbers() and print_letters().These functions will be executed by separate threads.
//2. In the main() function, we create two threads : thread1 and thread2.We pass the print_numbers() and print_letters() functions as arguments to the thread constructors.
//3. We start the threads using the thread1.join() and thread2.join() statements.This allows the threads to execute concurrently.
//4. The print_numbers() function prints numbers from 0 to 9, pausing for 1 second between each print statement.
//5. The print_letters() function prints letters from 'a' to 'j', pausing for 1 second between each print statement.
//6. Because the threads are running concurrently, the output will be interleaved.You'll see numbers and letters printed in an alternating pattern.
//
//The key benefits of multithreading in this example are :
//
//-Improved responsiveness : The program can perform multiple tasks at the same time, making it more responsive to the user.
//- Increased throughput : By executing multiple tasks concurrently, the program can complete tasks more quickly*/
//
//




void print_numbers() {
    for (int i = 0; i < 10; i++) {
        std::cout << i << std::endl;
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }
}

void print_letters() {
    for (char letter = 'a'; letter <= 'j'; letter++) {
        std::cout << letter << std::endl;
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }
}

int main() {
    // Create threads
    std::thread thread1(print_numbers);
    std::thread thread2(print_letters);

    // Wait for both threads to finish
    thread1.join();
    thread2.join();

    return 0;
}

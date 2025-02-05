//Here is an example for Lesson 7:


#include <iostream>
#include <thread>
#include <mutex>
#include <chrono>

std::mutex mtx;

void calculate_sum(int* arr, int size, int id) {
    int sum = 0;
    for (int i = 0; i < size; i++) {
        sum += arr[i];
    }
    mtx.lock();
    std::cout << "Thread " << id << ": Sum of array is " << sum << std::endl;
    mtx.unlock();
}

void calculate_product(int* arr, int size, int id) {
    int product = 1;
    for (int i = 0; i < size; i++) {
        product *= arr[i];
    }
    mtx.lock();
    std::cout << "Thread " << id << ": Product of array is " << product << std::endl;
    mtx.unlock();
}

int main() {
    int arr[] = { 1, 2, 3, 4, 5 };
    int size = sizeof(arr) / sizeof(arr[0]);

    // Create threads
    std::thread sumThread(calculate_sum, arr, size, 1);
    std::thread productThread(calculate_product, arr, size, 2);

    // Join threads
    sumThread.join();
    productThread.join();

    return 0;
}


//In this example, we have two threads that perform different calculations on the 
//same array.The calculate_sum function calculates the sum of the array elements, 
//and the calculate_product function calculates the product of the array elements.
//The results are printed to the console in a thread - safe manner using a mutex.
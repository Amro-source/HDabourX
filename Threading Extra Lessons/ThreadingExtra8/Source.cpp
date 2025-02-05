//Here's an example for Lesson 8:


#include <iostream>
#include <thread>
#include <mutex>
#include <vector>

std::mutex mtx;
int shared_sum = 0;

void partial_sum(const std::vector<int>& arr, int start, int end) {
    int local_sum = 0;
    for (int i = start; i < end; i++) {
        local_sum += arr[i];
    }
    mtx.lock();
    shared_sum += local_sum;
    mtx.unlock();
}

int main() {
    std::vector<int> arr(1000000);
    for (int i = 0; i < 1000000; i++) {
        arr[i] = i;
    }

    int num_threads = 4;
    std::vector<std::thread> threads;

    int chunk_size = arr.size() / num_threads;

    for (int i = 0; i < num_threads; i++) {
        int start = i * chunk_size;
        int end = (i == num_threads - 1) ? arr.size() : (i + 1) * chunk_size;
        threads.emplace_back(partial_sum, std::ref(arr), start, end);
    }

    for (auto& thread : threads) {
        thread.join();
    }

    std::cout << "Final sum: " << shared_sum << std::endl;

    return 0;
}


//
//In this example, we have a large array of integers and we want to calculate the sum of all 
//elements in parallel using multiple threads.
//
//We divide the array into chunks and assign each chunk to a separate thread.Each thread 
//calculates the sum of its chunk and updates a shared variable shared_sum using a mutex
//to ensure thread safety.
//
//Finally, we join all threads and print the final sum.
//
//This example demonstrates how multiple threads can cooperate to achieve a common goal, 
//which is a fundamental concept in parallel programming.
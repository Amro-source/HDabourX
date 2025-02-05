//Here's the modified code:


#include <iostream>
#include <thread>
#include <mutex>
#include <vector>
#include <chrono>

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

int calculate_without_threads(const std::vector<int>& arr) {
    int sum = 0;
    for (int i = 0; i < arr.size(); i++) {
        sum += arr[i];
    }
    return sum;
}

int main() {
    std::vector<int> arr(1000000);
    for (int i = 0; i < 1000000; i++) {
        arr[i] = i;
    }

    int num_threads = 4;

    // Calculate without threads
    auto start_time = std::chrono::high_resolution_clock::now();
    int result_without_threads = calculate_without_threads(arr);
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration_without_threads = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    std::cout << "Result without threads: " << result_without_threads << std::endl;
    std::cout << "Time taken without threads: " << duration_without_threads.count() << " milliseconds" << std::endl;

    // Calculate with threads
    shared_sum = 0;
    std::vector<std::thread> threads;
    int chunk_size = arr.size() / num_threads;
    start_time = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < num_threads; i++) {
        int start = i * chunk_size;
        int end = (i == num_threads - 1) ? arr.size() : (i + 1) * chunk_size;
        threads.emplace_back(partial_sum, std::ref(arr), start, end);
    }
    for (auto& thread : threads) {
        thread.join();
    }
    end_time = std::chrono::high_resolution_clock::now();
    auto duration_with_threads = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    std::cout << "Result with threads: " << shared_sum << std::endl;
    std::cout << "Time taken with threads: " << duration_with_threads.count() << " milliseconds" << std::endl;

    return 0;
}

//
//This code calculates the sum of the array elements using both a single - threaded approach
//and a multi - threaded approach.It also measures the time taken by each approach using 
//the std::chrono library.
//
//Note that the results may vary depending on the system's hardware and software configuration.
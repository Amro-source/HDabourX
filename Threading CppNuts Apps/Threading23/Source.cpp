#include <iostream>
#include <thread>
#include <vector>
#include <cassert>

using namespace std;

void worker(int number) {
    cout << "I am Worker Thread No : " << number << endl;
}

int main() {
    vector<std::thread> workers;

    unsigned long const hardware_threads = std::thread::hardware_concurrency();
    unsigned long thread_count = (hardware_threads > 0) ? hardware_threads : 4;  // Default to 4 if unavailable

    std::cout << "Number of hardware threads: " << thread_count << std::endl;

    for (int i = 0; i < thread_count; ++i) {
        workers.emplace_back(worker, i);
    }

    // Correct way to join threads
    for (auto& t : workers) {
        if (t.joinable()) {
            cout << "Joining thread " << t.get_id() << endl;
            t.join();
        }
    }

    return 0;
}

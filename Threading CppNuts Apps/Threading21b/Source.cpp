#include <iostream>
#include <thread>
#include <vector>
#include <chrono>

using namespace std;

void worker(int number) {
    cout << "I am Worker Thread No : " << number << endl;
}

int main() {
    vector<std::thread> workers;

    for (int i = 0; i < 3; ++i) {
        workers.emplace_back(worker, i);
    }

    // Manually join all threads before exiting
    for (auto& t : workers) {
        if (t.joinable()) {
            t.join();
        }
    }

    cout << "Hello There" << endl;
    return 0;
}

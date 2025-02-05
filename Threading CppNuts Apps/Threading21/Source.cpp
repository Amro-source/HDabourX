#include <iostream>
#include <thread>
#include <chrono>
#include <vector>
#include <cassert>
#include <algorithm>

using namespace std;

void worker(int number) {
    cout << "I am Worker Thread No : " << number << endl;
}

int main() {
    vector<std::jthread> workers;

    for (int i = 0; i < 3; ++i) {
        workers.emplace_back(worker, i);  // `emplace_back` constructs in-place, avoiding extra moves
    }

    cout << "Hello There" << endl;

    // jthreads automatically join on scope exit, ensuring all worker threads complete before main exits.
    return 0;
}

#include <iostream>
#include <thread>
#include <mutex>

using namespace std;

int myAmount = 0;
std::mutex m;

void addMoney() {
    std::lock_guard<std::mutex> lock(m); // Fix: lock_guard instead of lock_gaurd
    ++myAmount;
}

int main() {
    std::thread t1(addMoney);
    std::thread t2(addMoney);
    t1.join();
    t2.join();
    cout << myAmount << endl;
    return 0;
}

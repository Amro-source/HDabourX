#include <iostream>
#include <thread>
#include <mutex>
#include <condition_variable>

using namespace std;

int myAmount = 0;
std::mutex m;
std::condition_variable cv;
long balance = 0;

void addMoney(int money) {
    std::lock_guard<std::mutex> lg(m); // Fix: lock_guard instead of lock_gaurd
    balance += money;
    cv.notify_one();
}

void withdrawMoney(int money)
{
    std::unique_lock<std::mutex> ul(m); // Fix: lock_guard instead of lock_gaurd

    cv.wait(ul, [] { return (balance != 0) ? true : false;     });

    if (balance >= money)
    { 
        balance -= money;
        cout << "Amount DEducted :" << money << endl;


    }
    else {
        cout << "Amount can't be deducted.Current balance is less than  " <<money  << endl;


    }


    cout << "Current Balance is " << balance << endl;
 
}




int main() {
    std::thread t1(withdrawMoney,500);
    this_thread::sleep_for(chrono::seconds(2));
    std::thread t2(addMoney,500);
    t1.join();
    t2.join();
    cout << myAmount << endl;
    return 0;
}

// TOPIC: Different Types Of Thread Creation And Calling.
// There are 5 different types of creating threads and calling callable in threads.

#include <iostream>	
#include <thread>
using namespace std;

// 1. Function Pointer
// this is the very basic form of creating threads.

void fun(int x) {
	while (x-- > 0) {
		cout << x << endl;
	}
}

int main() {
	std::thread t(fun, 10);
	t.join();
	return 0;
}


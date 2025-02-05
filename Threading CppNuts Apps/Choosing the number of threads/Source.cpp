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

//
//You're setting up a vector of worker threads and querying the number of hardware threads available on your system.
//
//Here's a breakdown of what you're doing :
//
//1. std::thread::hardware_concurrency() : This function returns the number of hardware threads available on your system.This is typically the number of CPU cores or hyper - threading units available.
//2. unsigned long const hardware_threads = ... : You're storing the result of the above function call in a constant variable named hardware_threads.
//3. vector<std::thread> workers; : You're declaring a vector to store std::thread objects, which will represent your worker threads.
//
//By using std::thread::hardware_concurrency(), you're ensuring that you're not creating more threads than your system can handle efficiently.This is important because creating too many threads can lead to performance degradation due to context switching and other overheads.
//
//Next, you'll likely want to create and launch your worker threads, passing them some work to do. You can do this using a loop that creates and starts each thread, like so:
//
//for (unsigned long i = 0; i < hardware_threads; ++i) {
//	workers.emplace_back([i] { /* thread work goes here */ });
//}
//
//Replace the comment with the actual work you want each thread to perform.


int main() {

	vector<std::thread> workers;

	unsigned long const hardware_threads = std::thread::hardware_concurrency();

	std::cout << hardware_threads << endl;

	for (int i = 0; i < hardware_threads; ++i) {
		workers.emplace_back(std::thread(worker, i));
	}

	std::for_each(workers.begin(), workers.end(), [](std::thread& t) {
		assert(t.joinable());
		t.join();
		});
	return 0;
}
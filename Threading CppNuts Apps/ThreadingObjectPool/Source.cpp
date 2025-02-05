#include <thread>
#include <vector>
#include <iostream>
#include <queue>
#include <thread>
#include <functional>
#include <sstream>
#include <mutex>
#include <condition_variable>

std::string get_thread_id() {

	auto myid = std::this_thread::get_id();
	std::stringstream ss;
	ss << myid;

	std::string mystr = ss.str();
	return mystr;



}


class Object {
public:
	int _num;

	Object(int num) : _num {num} {}

	void doSomething() { std::cout << "Object doing Something " << std::endl; }




};



class ObjectPool {

public :
	std::queue<std::shared_ptr<Object>> pool;

	int poolSize;

	std::mutex mutex_;

	ObjectPool(int size) :poolSize(size) {
		std::cout << "Object Pool Constructed " << std::endl;

		for (int i = 0;i < poolSize;i++)
		{
			std::cout << "Object" << i <<std::endl;

			pool.push(std::make_shared<Object> (Object(i)));


		}



	}


	std::shared_ptr<Object> acquireObject() {

		std::unique_lock<std::mutex>  lock(mutex_);

		if (!pool.empty()) {

			auto obj = pool.front();
			printf("Thread %s is acquired object %d ", get_thread_id().c_str(), obj->_num); 

			pool.pop();

			return obj;
		}


		return nullptr;

	}
	void releaseObject(std::shared_ptr<Object> obj) {

		std::unique_lock<std::mutex>  lock(mutex_);
		
		printf("Thread %s is released object %d ", get_thread_id().c_str(), obj->_num);
		pool.push(obj);

	}





};


void worker(ObjectPool & pool) {

	for (int i = 0;i < 15;i++)
	{

		auto obj = pool.acquireObject();

		if (obj)
		{
			int worked_for = rand() % 1000;

			std::this_thread::sleep_for(std::chrono::microseconds(worked_for));

			pool.releaseObject(obj);
		}
		else {
			printf("Thread %s failed to acquire an object %d ", get_thread_id().c_str(), obj->_num);

		}


	}



}


int main()

{
	ObjectPool pool(3);
	// Create a pool with a maximum size of 3 
	
	std::vector<std::thread> workers; 
	int workers_count = 5;

	for (int i = 0; i < workers_count; i++) {
		workers.emplace_back(worker, std::ref(pool));
	}

	for (int i = 0; i < workers_count; i++) { 
	
		workers[i].join(); 
	
	}

	return 0;

}


//bi int worked_tor = rand() % 1000; 63 
//
//std;:this_thread::sleep_for(std::chrono:milliseconds(worked_for)); 64 pool.releaseObject(obj); 65 }
// else { 66 printf("Thread %s failed to acquire object\n", get_thread_id().c_str()); 67 } 68 } 691 70 71 int main() {
//	 72 ObjectPool pool(3); // Workers list will store all the threads. 74  // How many threads you want ? 
//	 75 76 
//		 7 1 79 80  83 84 return 0; 85
//	 } 86
//
//		 •
//

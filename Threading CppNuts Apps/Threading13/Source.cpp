#include <iostream>
#include <thread>
#include <mutex>
#include <deque>
#include <vector>
#include <condition_variable>

using namespace std;

class Semaphore {
public:
    void insert(const std::vector<int>& Vec) {
        int freeId = down();  // Get a free buffer index

        if (freeId != -1) {
            {
                std::lock_guard<std::mutex> lock(m[freeId]);
                VecOfQue[freeId].insert(VecOfQue[freeId].end(), Vec.begin(), Vec.end());
                cout << "Inserted into queue " << freeId << endl;
            }
            up(freeId);  // Signal availability
        }
    }

    void remove() {
        while (true) {  // Keep removing elements
            std::unique_lock<std::mutex> lock(headMutex);
            int bufferId = getNonEmptyBufferId();

            if (bufferId == -1) {
                cvEmpty.wait(lock, [this] { return !allBuffersEmpty(); });
                continue;
            }

            lock.unlock();  // Unlock to allow insertions while removing

            {
                std::lock_guard<std::mutex> bufferLock(m[bufferId]);
                if (!VecOfQue[bufferId].empty()) {
                    cout << "Removed " << VecOfQue[bufferId].front() << " from queue " << bufferId << endl;
                    VecOfQue[bufferId].pop_front();
                }
            }
        }
    }

private:
    int down() {
        std::unique_lock<std::mutex> lock(headMutex);
        int freeId = getFreeBufferId();

        if (freeId == -1) {
            cvFull.wait(lock, [this] { return getFreeBufferId() != -1; });
            freeId = getFreeBufferId();
        }

        return freeId;
    }

    void up(int bufferId) {
        cvFull.notify_one();
        cvEmpty.notify_one();
    }

    int getFreeBufferId() {
        for (int i = 0; i < 4; ++i) {
            if (VecOfQue[i].empty()) return i;
        }
        return -1;
    }

    int getNonEmptyBufferId() {
        for (int i = 0; i < 4; ++i) {
            if (!VecOfQue[i].empty()) return i;
        }
        return -1;
    }

    bool allBuffersEmpty() {
        for (int i = 0; i < 4; ++i) {
            if (!VecOfQue[i].empty()) return false;
        }
        return true;
    }

private:
    std::mutex headMutex;
    std::mutex m[4];
    std::condition_variable cvFull, cvEmpty;
    deque<int> VecOfQue[4];
} sem;

int main() {
    std::thread con(&Semaphore::remove, &sem);
    std::vector<std::thread> threadVec;

    for (int i = 0; i < 10; ++i) {
        std::vector<int> job = { i * 1, i * 2, i * 3, i * 4 };
        threadVec.push_back(std::thread(&Semaphore::insert, &sem, std::ref(job)));
    }

    for (auto& thread : threadVec) {
        thread.join();
    }

    con.join();
    return 0;
}

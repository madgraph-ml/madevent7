#pragma once

#include <mutex>
#include <algorithm>
#include <vector>
#include <thread>
#include <condition_variable>
#include <functional>

namespace madevent {
namespace cpu {

class ThreadPool {
public:
    static void set_thread_count(int new_count);

    static ThreadPool& instance() {
        static ThreadPool instance;
        return instance;
    }

    ~ThreadPool();

    template<typename F>
    void parallel_for(F func, std::size_t count) {
        if (thread_count == 0 || count < thread_count * 100) {
            for (std::size_t i = 0; i < count; ++i) {
                func(i);
            }
        } else {
            std::unique_lock<std::mutex> lock(mutex);
            count_per_thread = (count + thread_count - 1) / thread_count;
            total_count = count;
            busy_threads = thread_count;
            std::fill(thread_done.begin(), thread_done.end(), false);
            job = [&](std::size_t start_index, std::size_t stop_index) {
                for (std::size_t i = start_index; i < stop_index; ++i) {
                    func(i);
                }
            };
            lock.unlock();
            cv_run.notify_all();
            lock.lock();
            cv_done.wait(lock, [&]{ return busy_threads == 0; });
        }
    }

    ThreadPool(ThreadPool const&) = delete;
    void operator=(ThreadPool const&) = delete;

private:
    static inline int thread_count = -1;

    ThreadPool();
    void thread_loop(int index);
    void adjust_thread_count();
    std::mutex mutex;
    std::condition_variable cv_run, cv_done;
    std::vector<std::thread> threads;
    std::optional<std::function<void(std::size_t, std::size_t)>> job;
    std::vector<bool> thread_done;
    std::size_t total_count, count_per_thread, busy_threads;
};

}
}

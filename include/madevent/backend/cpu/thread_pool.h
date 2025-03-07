#pragma once

#include <mutex>
#include <algorithm>
#include <vector>
#include <thread>
#include <condition_variable>
#include <functional>
#include <random>

namespace madevent {
namespace cpu {

class ThreadPool {
public:
    static void set_thread_count(int new_count);
    static const bool pass_thread_id = true;

    static ThreadPool& instance() {
        static ThreadPool instance;
        return instance;
    }

    ~ThreadPool();

    template<typename F>
    void parallel(F func) {
        std::unique_lock<std::mutex> lock(mutex);
        count_per_thread = 1;
        total_count = thread_count;
        busy_threads = thread_count;
        std::fill(thread_done.begin(), thread_done.end(), false);
        job = [&](std::size_t start_index, std::size_t stop_index, int thread_id) {
            func(start_index);
        };
        lock.unlock();
        cv_run.notify_all();
        lock.lock();
        cv_done.wait(lock, [&]{ return busy_threads == 0; });
    }

    template<bool _pass_thread_id=false, typename F>
    void parallel_for(F func, std::size_t count) {
        if (thread_count == 0 || count < thread_count * 100) {
            for (std::size_t i = 0; i < count; ++i) {
                if constexpr (_pass_thread_id) {
                    func(i, 0);
                } else {
                    func(i);
                }
            }
        } else {
            std::unique_lock<std::mutex> lock(mutex);
            count_per_thread = (count + thread_count - 1) / thread_count;
            total_count = count;
            busy_threads = thread_count;
            std::fill(thread_done.begin(), thread_done.end(), false);
            job = [&](std::size_t start_index, std::size_t stop_index, int thread_id) {
                for (std::size_t i = start_index; i < stop_index; ++i) {
                    if constexpr (_pass_thread_id) {
                        func(i, thread_id);
                    } else {
                        func(i);
                    }
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
    std::size_t get_thread_count() const { return threads.size(); }
    double random(int thread_index) {
        return rand_dist(thread_rand_gens[thread_index]);
    }

private:
    static inline int thread_count = -1;

    ThreadPool();
    void thread_loop(int index);
    void adjust_thread_count();
    std::mutex mutex;
    std::condition_variable cv_run, cv_done;
    std::vector<std::thread> threads;
    std::optional<std::function<void(std::size_t, std::size_t, int)>> job;
    std::vector<bool> thread_done;
    std::size_t total_count, count_per_thread, busy_threads;
    std::random_device rand_device;
    std::uniform_real_distribution<double> rand_dist;
    std::vector<std::mt19937> thread_rand_gens;
};

}
}

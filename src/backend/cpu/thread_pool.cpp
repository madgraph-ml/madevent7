#include "madevent/backend/cpu/thread_pool.h"
#include <iostream>

using namespace madevent::cpu;

void ThreadPool::set_thread_count(int new_count) {
    if (new_count == -1) {
        thread_count = std::thread::hardware_concurrency();
    } else if (new_count == 1) {
        thread_count = 0;
    } else if (new_count < -1) {
        throw std::invalid_argument("thread count must be -1 or larger");
    } else {
        thread_count = new_count;
    }
    instance().adjust_thread_count();
}

ThreadPool::ThreadPool() {
    if (thread_count == -1) {
        thread_count = std::thread::hardware_concurrency();
    }
    for (int i = 0; i < thread_count; ++i) {
        thread_done.push_back(true);
        thread_rand_gens.emplace_back(rand_device());
        threads.emplace_back(&ThreadPool::thread_loop, this, i);
    }
}

ThreadPool::~ThreadPool() {
    thread_count = 0;
    adjust_thread_count();
}

void ThreadPool::thread_loop(int index) {
    std::unique_lock<std::mutex> lock(mutex);
    while (true) {
        cv_run.wait(lock, [&]{
            if (index < 0 || index >= thread_done.size()) {
                std::cout << "thread error index=" << index << " len=" << thread_done.size() << "\n";
            }
            return !thread_done.at(index);
        });
        lock.unlock();

        if (job) {
            std::size_t start_index = index * count_per_thread;
            std::size_t stop_index = std::min(total_count, start_index + count_per_thread);
            (*job)(start_index, stop_index, index);
        } else {
            return;
        }

        lock.lock();
        --busy_threads;
        thread_done[index] = true;
        cv_done.notify_one();
    }
}

void ThreadPool::adjust_thread_count() {
    while (thread_rand_gens.size() < std::max(thread_count, 1)) {
        thread_rand_gens.emplace_back(rand_device());
    }
    if (threads.size() < thread_count) {
        for (int i = threads.size(); i < thread_count; ++i) {
            thread_done.push_back(true);
            threads.emplace_back(&ThreadPool::thread_loop, this, i);
        }
    } else if (threads.size() > thread_count) {
        {
            std::unique_lock<std::mutex> lock(mutex);
            std::fill(thread_done.begin() + thread_count, thread_done.end(), false);
            job.reset();
        }
        cv_run.notify_all();
        std::for_each(threads.begin() + thread_count, threads.end(), [](auto& thread) {
            thread.join();
        });
        thread_done.erase(thread_done.begin() + thread_count, thread_done.end());
        threads.erase(threads.begin() + thread_count, threads.end());
    }
}

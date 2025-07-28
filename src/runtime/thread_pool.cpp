#include "madevent/runtime/thread_pool.h"

using namespace madevent;

ThreadPool::ThreadPool(int thread_count) {
    set_thread_count(thread_count);
}

ThreadPool::~ThreadPool() {
    set_thread_count(0);
}

void ThreadPool::set_thread_count(int new_count) {
    {
        std::unique_lock<std::mutex> lock(_mutex);
        _thread_count = new_count < 0 ? std::thread::hardware_concurrency() : new_count;
    }

    if (_threads.size() < _thread_count) {
        for (int i = _threads.size(); i < _thread_count; ++i) {
            _threads.emplace_back(&ThreadPool::thread_loop, this, i);
        }
    } else if (_threads.size() > _thread_count) {
        _cv_run.notify_all();
        std::for_each(_threads.begin() + _thread_count, _threads.end(), [](auto& thread) {
            thread.join();
        });
        _threads.erase(_threads.begin() + _thread_count, _threads.end());
    }

    for (auto& [id, listener] : _listeners) listener(_thread_count);
}

void ThreadPool::submit(std::function<int(std::size_t)> job) {
    std::unique_lock<std::mutex> lock(_mutex);
    _job_queue.push(job);
    _cv_run.notify_one();
}

std::optional<int> ThreadPool::wait() {
    std::unique_lock<std::mutex> lock(_mutex);
    if (_done_queue.empty()) {
        if (_job_queue.empty() && _busy_threads == 0) return std::nullopt;
        _cv_done.wait(lock, [&]{ return !_done_queue.empty(); });
    }
    int result = _done_queue.front();
    _done_queue.pop();
    return result;
}

std::size_t ThreadPool::add_listener(std::function<void(std::size_t)> listener) {
    _listeners[_listener_id] = listener;
    return _listener_id++;
}

void ThreadPool::remove_listener(std::size_t id) {
    if (auto search = _listeners.find(id); search != _listeners.end()) {
        _listeners.erase(search);
    } else {
        throw std::invalid_argument("Listener id not found");
    }
}

void ThreadPool::thread_loop(std::size_t index) {
    std::unique_lock<std::mutex> lock(_mutex);
    while (true) {
        _cv_run.wait(lock, [&]{ return !_job_queue.empty() || index >= _thread_count; });
        if (index >= _thread_count) return;
        auto job = _job_queue.front();
        _job_queue.pop();
        ++_busy_threads;
        lock.unlock();
        int result = job(index);
        lock.lock();
        --_busy_threads;
        _done_queue.push(result);
        _cv_done.notify_one();
    }
}

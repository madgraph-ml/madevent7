#include "madevent/runtime/thread_pool.h"

#include "madevent/util.h"

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

void ThreadPool::submit(JobFunc job) {
    if (_buffer_submit) {
        _job_buffer.push_back(job);
    } else {
        std::unique_lock<std::mutex> lock(_mutex);
        _job_queue.push_back(job);
        if (!_done_queue.empty()) {
            _done_buffer.insert(_done_buffer.begin(), _done_queue.rbegin(), _done_queue.rend());
            _done_queue.clear();
        }
        lock.unlock();
        _cv_run.notify_one();
    }
}

void ThreadPool::submit_all() {
    _buffer_submit = false;
    if (_job_buffer.empty()) return;
    std::unique_lock<std::mutex> lock(_mutex);
    for (auto& job : _job_buffer) _job_queue.push_back(std::move(job));
    if (!_done_queue.empty()) {
        _done_buffer.insert(_done_buffer.begin(), _done_queue.rbegin(), _done_queue.rend());
        _done_queue.clear();
    }
    lock.unlock();
    _cv_run.notify_all();
    _job_buffer.clear();
}

bool ThreadPool::fill_done_buffer() {
    if (!_done_buffer.empty()) return true;

    std::unique_lock<std::mutex> lock(_mutex);
    if (_done_queue.empty()) {
        if (_job_queue.empty() && _busy_threads == 0) return false;
        _cv_done.wait(lock, [&]{ return !_done_queue.empty(); });
    }
    _done_buffer.insert(_done_buffer.begin(), _done_queue.rbegin(), _done_queue.rend());
    _done_queue.clear();
    return true;
}

std::optional<std::size_t> ThreadPool::wait() {
    if (!fill_done_buffer()) return std::nullopt;
    std::size_t result = _done_buffer.back();
    _done_buffer.pop_back();
    return result;
}

std::vector<std::size_t> ThreadPool::wait_multiple() {
    if (!fill_done_buffer()) return {};
    std::vector<std::size_t> ret(_done_buffer.rbegin(), _done_buffer.rend());
    _done_buffer.clear();
    return ret;
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
    _thread_index = index;
    std::unique_lock<std::mutex> lock(_mutex);
    while (true) {
        _cv_run.wait(lock, [&]{ return !_job_queue.empty() || index >= _thread_count; });
        if (index >= _thread_count) return;
        auto job = _job_queue.front();
        _job_queue.pop_front();
        ++_busy_threads;
        lock.unlock();
        std::size_t result = job();
        lock.lock();
        --_busy_threads;
        _done_queue.push_back(result);
        _cv_done.notify_one();
    }
}

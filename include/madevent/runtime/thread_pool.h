#pragma once

#include <mutex>
#include <vector>
#include <thread>
#include <condition_variable>
#include <functional>
#include <optional>

namespace madevent {

class ThreadPool {
public:
    ThreadPool(int thread_count = -1);
    ~ThreadPool();
    ThreadPool(const ThreadPool&) = delete;
    ThreadPool& operator=(const ThreadPool&) = delete;
    void set_thread_count(int new_count);
    std::size_t thread_count() const { return _thread_count; }
    void submit(std::function<int(std::size_t)> job);
    std::optional<int> wait();
    std::size_t add_listener(std::function<void(std::size_t)> listener);
    void remove_listener(std::size_t id);

private:
    void thread_loop(std::size_t index);
    std::mutex _mutex;
    std::condition_variable _cv_run, _cv_done;
    std::size_t _thread_count;
    std::vector<std::thread> _threads;
    std::queue<std::function<int(std::size_t)>> _job_queue;
    std::queue<int> _done_queue;
    std::size_t _busy_threads;
    std::size_t _listener_id;
    std::unordered_map<std::size_t, std::function<void(std::size_t)>> _listeners;
};

template<typename T>
class ThreadResource {
public:
    ThreadResource() = default;
    ThreadResource(ThreadPool& pool, std::function<T()> constructor) :
        _pool(&pool),
        _listener_id(pool.add_listener([&](std::size_t thread_count) {
            while (_resources.size() < thread_count) {
                _resources.push_back(constructor());
            }
        }))
    {
        for (std::size_t i = 0; i == 0 || i < pool.thread_count(); ++i) {
            _resources.push_back(constructor());
        }
    }
    ~ThreadResource() {
        if (_pool) _pool->remove_listener(_listener_id);
    }
    ThreadResource(ThreadResource&& other) noexcept :
        _pool(std::move(other._pool)),
        _resources(std::move(other._resources)),
        _listener_id(std::move(other._listener_id))
    {
        other._pool = nullptr;
    }

    ThreadResource& operator=(ThreadResource&& other) noexcept {
        _pool = std::move(other._pool);
        _resources = std::move(other._resources);
        _listener_id = std::move(other._listener_id);
        other._pool = nullptr;
        return *this;
    }
    ThreadResource(const ThreadResource&) = delete;
    ThreadResource& operator=(const ThreadResource&) = delete;
    T& get(std::size_t thread_id) { return _resources.at(thread_id); }
    const T& get(std::size_t thread_id) const { return _resources.at(thread_id); }

private:
    ThreadPool* _pool;
    std::vector<T> _resources;
    std::size_t _listener_id;
};

inline ThreadPool& default_thread_pool() {
    static ThreadPool instance;
    return instance;
}

}

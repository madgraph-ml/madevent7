#pragma once

#include <mutex>
#include <vector>
#include <thread>
#include <condition_variable>
#include <functional>
#include <optional>
#include <deque>

namespace madevent {

class ThreadPool {
public:
    using JobFunc = std::function<std::size_t()>;
    ThreadPool(int thread_count = -1);
    ~ThreadPool();
    ThreadPool(const ThreadPool&) = delete;
    ThreadPool& operator=(const ThreadPool&) = delete;
    void set_thread_count(int new_count);
    std::size_t thread_count() const { return _thread_count; }
    void submit(JobFunc job);
    void submit(std::vector<JobFunc>& jobs);
    std::optional<std::size_t> wait();
    std::vector<std::size_t> wait_multiple();
    std::size_t add_listener(std::function<void(std::size_t)> listener);
    void remove_listener(std::size_t id);

    static std::size_t thread_index() { return _thread_index; }

private:
    static inline thread_local std::size_t _thread_index = 0;
    static const std::size_t QUEUE_SIZE_PER_THREAD = 16384;

    void thread_loop(std::size_t index);
    bool fill_done_cache();

    std::mutex _mutex;
    std::condition_variable _cv_run, _cv_done;
    std::size_t _thread_count;
    std::vector<std::thread> _threads;
    std::size_t _queue_mask;
    std::vector<JobFunc> _job_queue;
    std::atomic<std::size_t> _job_queue_begin;
    std::atomic<std::size_t> _job_queue_end;
    std::atomic<std::size_t> _job_queue_read;
    std::deque<std::size_t> _done_queue_cache;
    std::vector<std::size_t> _done_queue;
    std::atomic<std::size_t> _done_queue_begin;
    std::atomic<std::size_t> _done_queue_end;
    std::atomic<std::size_t> _done_queue_write;
    std::size_t _busy_threads;
    std::size_t _listener_id = 0;
    std::unordered_map<std::size_t, std::function<void(std::size_t)>> _listeners;
    bool _buffer_submit;
};

template<typename T>
class ThreadResource {
public:
    ThreadResource() = default;
    ThreadResource(
        ThreadPool& pool,
        std::function<T()> constructor,
        std::optional<std::function<void(T&)>> destructor = std::nullopt
    ) :
        _pool(&pool),
        _listener_id(pool.add_listener([&](std::size_t thread_count) {
            while (_resources.size() < thread_count) {
                _resources.push_back(constructor());
            }
        })),
        _destructor(destructor)
    {
        for (std::size_t i = 0; i == 0 || i < pool.thread_count(); ++i) {
            _resources.push_back(constructor());
        }
    }
    ~ThreadResource() {
        if (_destructor) {
            for (auto& res : _resources) (*_destructor)(res);
        }
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
    ThreadPool* _pool = nullptr;
    std::vector<T> _resources;
    std::size_t _listener_id;
    std::optional<std::function<void(T&)>> _destructor;
};

inline ThreadPool& default_thread_pool() {
    static ThreadPool instance;
    return instance;
}

}

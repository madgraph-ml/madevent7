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
        if (_busy_threads > 0) {
            throw std::runtime_error(
                "Cannot change thread count while thread pool is busy"
            );
        }
        _thread_count = new_count < 0 ? std::thread::hardware_concurrency() : new_count;

        std::size_t queue_size = _thread_count * QUEUE_SIZE_PER_THREAD;
        std::size_t queue_size_rounded = 1;
        while (queue_size_rounded < queue_size) queue_size_rounded *= 2;
        _job_queue.resize(queue_size_rounded);
        _done_queue.resize(queue_size_rounded);
        _queue_mask = queue_size_rounded - 1;
        _job_queue_begin = 0;
        _job_queue_end = 0;
        _job_queue_read = 0;
        _done_queue_begin = 0;
        _done_queue_end = 0;
        _done_queue_write = 0;
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
    // Spin until there is space in the queue. The queue should be sufficiently large
    // such that this never happens.
    std::size_t begin_index, current_index, next_index;
    do {
        begin_index = _job_queue_begin.load();
        current_index = _job_queue_end.load();
        next_index = (current_index + 1) & _queue_mask;
    } while(next_index == begin_index);
    _job_queue[current_index] = std::move(job);
    _job_queue_end.store(next_index);
    _cv_run.notify_one();
}

void ThreadPool::submit(std::vector<JobFunc>& jobs) {
    std::size_t job_index = 0;
    while (job_index < jobs.size()) {
        std::size_t begin_index, current_index;
        do {
            begin_index = _job_queue_begin.load();
            current_index = _job_queue_end.load();
        } while(((current_index + 1) & _queue_mask) == begin_index);
        for (;
            ((current_index + 1) & _queue_mask) != begin_index && job_index < jobs.size();
            ++job_index, current_index = (current_index + 1) & _queue_mask
        ) {
            _job_queue[current_index] = std::move(jobs[job_index]);
        }
        _job_queue_end.store(current_index);
        _cv_run.notify_all();
    }

}

bool ThreadPool::fill_done_cache() {
    if (!_done_queue_cache.empty()) return true;

    std::size_t begin_index = _done_queue_begin.load();
    std::size_t end_index = _done_queue_end.load();
    if (begin_index == end_index) {
        std::unique_lock<std::mutex> lock(_mutex);
        //if (_job_queue_begin == _job_queue_end && _busy_threads == 0) return false;
        bool no_more_jobs;
        _cv_done.wait(lock, [&]{
            end_index = _done_queue_end.load();
            no_more_jobs = false;
            if (_busy_threads == 0) {
                if (_job_queue_begin.load() == _job_queue_end.load()) {
                    no_more_jobs = true;
                } else {
                    //TODO: why is this necessary?
                    _cv_run.notify_all();
                }
            }
            return begin_index != end_index || no_more_jobs;
        });
        if (begin_index == end_index) {
            return false;
        }
    }

    for (std::size_t i = begin_index; i != end_index; i = (i + 1) & _queue_mask) {
        _done_queue_cache.push_back(_done_queue[i]);
    }
    _done_queue_begin.store(end_index);
    return true;
}

std::optional<std::size_t> ThreadPool::wait() {
    if (!fill_done_cache()) return std::nullopt;
    std::size_t result = _done_queue_cache.back();
    _done_queue_cache.pop_back();
    return result;
}

std::vector<std::size_t> ThreadPool::wait_multiple() {
    if (!fill_done_cache()) return {};
    std::vector<std::size_t> ret(_done_queue_cache.rbegin(), _done_queue_cache.rend());
    _done_queue_cache.clear();
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
        _cv_run.wait(lock, [&]{
            return _job_queue_begin != _job_queue_end || index >= _thread_count;
        });
        if (index >= _thread_count) return;
        std::size_t bt = ++_busy_threads;
        lock.unlock();

        while (true) {
            // if queue is empty, go to sleep
            std::size_t current_index = _job_queue_read.load();
            if (current_index == _job_queue_end.load()) break;

            // spin until read index was increased
            std::size_t next_index = (current_index + 1) & _queue_mask;
            if (!_job_queue_read.compare_exchange_weak(current_index, next_index)) {
                continue;
            }

            // we now have exclusive access to the queue item at current_index
            JobFunc job = std::move(_job_queue[current_index]);

            // free up space in the queue
            std::size_t current_index_before = current_index;
            while(!_job_queue_begin.compare_exchange_weak(current_index, next_index)) {
                current_index = current_index_before;
            }

            // run job
            std::size_t result = job();

            // spin until there is space in done queue and it was successfully increased
            // spinning can be prevented by making the queue large enough
            std::size_t current_done_index, next_done_index, done_begin_index;
            do {
                done_begin_index = _done_queue_begin.load();
                current_done_index = _done_queue_write.load();
                next_done_index = (current_done_index + 1) & _queue_mask;
                if (next_done_index == done_begin_index) continue;
            } while(!_done_queue_write.compare_exchange_weak(
                current_done_index, next_done_index
            ));

            // we now have exclusive access to the queue item at current_done_index
            _done_queue[current_done_index] = result;

            // increase end index to make result available
            std::size_t current_done_index_before = current_done_index;
            while(!_done_queue_end.compare_exchange_weak(
                current_done_index, next_done_index
            )) {
                current_done_index = current_done_index_before;
            }

            // if queue was empty, notify waiting threads
            if (current_done_index_before == done_begin_index) {
                _cv_done.notify_one();
            }
        }

        lock.lock();
        bt = --_busy_threads;
        _cv_done.notify_one();
    }
}

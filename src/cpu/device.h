#pragma once

#include "madevent/runtime/tensor.h"
#include "madevent/runtime/thread_pool.h"
#include "simd.h"

namespace madevent {
namespace cpu {

inline std::tuple<std::size_t, std::size_t> job_count_and_size(
    std::size_t batch_size, bool single_job = false
) {
    if (single_job) return {1, batch_size};

    std::size_t min_batch_size = 64;
    std::size_t thread_count = default_thread_pool().thread_count();
    std::size_t job_count = batch_size < thread_count * min_batch_size ?
        (batch_size + min_batch_size - 1) / min_batch_size :
        thread_count;
    std::size_t job_size = (batch_size + job_count - 1) / job_count;
    job_size = (job_size + simd_vec_size - 1) / simd_vec_size * simd_vec_size;
    return {job_count, job_size};
}

class CpuDevice : public Device {
public:
    static constexpr bool is_concurrent = false;

    void* allocate(std::size_t size) const override {
        return new std::byte[size];
    }

    void free(void* ptr) const override {
        delete[] static_cast<std::byte*>(ptr);
    }

    void memcpy(void* to, void* from, std::size_t size) const override {
        auto to_u8 = static_cast<std::byte*>(to);
        auto from_u8 = static_cast<std::byte*>(from);
        std::copy(from_u8, from_u8 + size, to_u8);
    }

    void tensor_copy(const Tensor& source, Tensor& target) const override;
    void tensor_zero(Tensor& tensor) const override;
    void tensor_add(const Tensor& source, Tensor& target) const override;
    void tensor_cpu(const Tensor& source, Tensor& target) const override {}
    DevicePtr device_ptr() const override { return &instance(); }

    template<typename F>
    void foreach(std::size_t batch_size, F func, bool single_job = false) const {
        func(batch_size, 0);
    }

    CpuDevice(const CpuDevice&) = delete;
    CpuDevice& operator=(CpuDevice&) = delete;
    static const CpuDevice& instance() {
        static CpuDevice device;
        return device;
    }

protected:
    CpuDevice() = default;
};

class AsyncCpuDevice : public CpuDevice {
public:
    static constexpr bool is_concurrent = true;

    AsyncCpuDevice(int instr_index, std::size_t& instr_job_count) :
        _instr_index(instr_index), _instr_job_count(instr_job_count) {}
    int add_jobs(int count) const {
        _instr_job_count += count;
        return _instr_index;
    }

    template<typename F>
    void foreach(std::size_t batch_size, F func, bool single_job = false) const {
        auto [job_count, job_size] = job_count_and_size(batch_size, single_job);
        int result = add_jobs(job_size);
        for (std::size_t i = 0; i < job_count; ++i) {
            std::size_t offset = i * job_size;
            std::size_t count = std::min(job_size, batch_size - offset);
            default_thread_pool().submit(
                [count, offset, func, result]() {
                    func(count, offset);
                    return result;
                }
            );
        }
    }

private:
    int _instr_index;
    std::size_t& _instr_job_count;
};

extern "C" DevicePtr get_device();

}
}

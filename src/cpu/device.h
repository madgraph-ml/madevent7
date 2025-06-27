#include "madevent/runtime/tensor.h"

namespace madevent {
namespace cpu {

class CpuDevice : public Device {
public:
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

    CpuDevice(const CpuDevice&) = delete;
    CpuDevice& operator=(CpuDevice&) = delete;
    static const CpuDevice& instance() {
        static CpuDevice device;
        return device;
    }

private:
    CpuDevice() = default;
};

extern "C" DevicePtr get_device();

}
}

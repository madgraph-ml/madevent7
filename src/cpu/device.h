#include "madevent/runtime/tensor.h"

namespace madevent_cpu {

class CpuDevice : public madevent::Device {
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

    void tensor_copy(const madevent::Tensor& source, madevent::Tensor& target) const override;
    void tensor_zero(madevent::Tensor& tensor) const override;
    void tensor_add(const madevent::Tensor& source, madevent::Tensor& target) const override;

    CpuDevice(const CpuDevice&) = delete;
    CpuDevice& operator=(CpuDevice&) = delete;
    static CpuDevice* instance() {
        static CpuDevice device;
        return &device;
    }

private:
    CpuDevice() {}
};

extern "C" madevent::DevicePtr get_device();

}

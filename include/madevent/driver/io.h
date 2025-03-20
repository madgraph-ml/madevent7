#pragma once

#include "madevent/backend/tensor.h"

#include <fstream>

namespace madevent {


Tensor load_tensor(const std::string& file);
void save_tensor(const std::string& file, Tensor tensor);

struct ParticleRecord {
    int64_t pid;
    double e;
    double px;
    double py;
    double pz;
};

struct EventRecord {
    double weight;
};

class EventBuffer {
public:
    static std::size_t size(std::size_t particle_count) {
        return sizeof(EventRecord) + particle_count * sizeof(ParticleRecord);
    }

    EventBuffer(std::size_t particle_count) :
        _particle_count(particle_count), _data(size(particle_count)) {}
    char* data() { return _data.data(); }
    const char* data() const { return _data.data(); }
    std::size_t size() const { return _data.size(); }
    std::span<ParticleRecord> particles() {
        return {
            reinterpret_cast<ParticleRecord*>(_data.data() + sizeof(EventRecord)),
            _particle_count
        };
    }
    std::span<const ParticleRecord> particles() const {
        return {
            reinterpret_cast<const ParticleRecord*>(_data.data() + sizeof(EventRecord)),
            _particle_count
        };
    }
    EventRecord& event() {
        return *reinterpret_cast<EventRecord*>(_data.data());
    }
    const EventRecord& event() const {
        return *reinterpret_cast<const EventRecord*>(_data.data());
    }
private:
    std::size_t _particle_count;
    std::vector<char> _data;
};

class EventFile {
public:
    enum Mode { create, append, load };
    EventFile(
        const std::string& file_name, std::size_t particle_count = 0, Mode mode = create
    );
    EventFile(EventFile&& other) noexcept = default;
    EventFile& operator=(EventFile&& other) noexcept = default;
    void write(const EventBuffer& event);
    bool read(EventBuffer& event);
    void seek(std::size_t index);
    std::size_t unweight(double max_weight, std::function<double()> random_generator);
    void clear();
    std::size_t particle_count() const { return _particle_count; }
    ~EventFile();
private:
    std::string _file_name;
    std::size_t _event_count;
    std::size_t _current_event;
    std::size_t _capacity;
    std::size_t _particle_count;
    std::size_t _shape_pos;
    std::fstream _file_stream;
    std::size_t _header_size;
    Mode _mode;
};

}

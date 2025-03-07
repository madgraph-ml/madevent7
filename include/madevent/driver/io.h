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
    std::vector<ParticleRecord> particles;
};

class EventReader {
public:
    EventReader(const std::string& file);
    bool read(EventRecord& event);
private:
    std::size_t event_count;
    std::size_t particle_count;
    std::ifstream file_stream;
};

class EventWriter {
public:
    EventWriter(const std::string& file, std::size_t particle_count);
    void write(const EventRecord& event);
    ~EventWriter();
private:
    std::size_t event_count;
    std::size_t particle_count;
    std::size_t shape_pos;
    std::ofstream file_stream;
};

}

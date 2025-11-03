#pragma once

#include <fstream>
#include <cstring>

#include "madevent/runtime/tensor.h"
#include "madevent/runtime/lhe_output.h"

namespace madevent {


Tensor load_tensor(const std::string& file);
void save_tensor(const std::string& file, Tensor tensor);

using FieldDescr = std::pair<std::string, std::string>;

struct ParticleRecord {
    static constexpr std::array<FieldDescr, 4> field_descr = {{
        {"energy", "<f8"}, {"px", "<f8"}, {"py", "<f8"}, {"pz", "<f8"}
    }};

    double energy;
    double px;
    double py;
    double pz;
};

struct EventRecord {
    static constexpr std::array<FieldDescr, 4> field_descr = {{
        {"diagram_index", "<i4"},
        {"color_index", "<i4"},
        {"flavor_index", "<i4"},
        {"helicity_index", "<i4"}
    }};

    int diagram_index;
    int color_index;
    int flavor_index;
    int helicity_index;
};

struct EmptyParticleRecord {
    static constexpr std::array<FieldDescr, 0> field_descr = {};
};

struct EventWeightRecord {
    static constexpr std::array<FieldDescr, 1> field_descr = {{
        {"weight", "<f8"},
    }};

    double weight;
};

struct PackedLHEParticle {
    static constexpr std::array<FieldDescr, 13> field_descr = {{
        {"pdg_id", "<i4"},
        {"status_code", "<i4"},
        {"mother1", "<i4"},
        {"mother2", "<i4"},
        {"color", "<i4"},
        {"anti_color", "<i4"},
        {"px", "<f8"},
        {"py", "<f8"},
        {"pz", "<f8"},
        {"energy", "<f8"},
        {"mass", "<f8"},
        {"lifetime", "<f8"},
        {"spin", "<f8"},
    }};

    PackedLHEParticle(const LHEParticle& particle) {
        std::memcpy(&data[0], &particle.pdg_id, sizeof(int));
        std::memcpy(&data[4], &particle.status_code, sizeof(int));
        std::memcpy(&data[8], &particle.mother1, sizeof(int));
        std::memcpy(&data[12], &particle.mother2, sizeof(int));
        std::memcpy(&data[16], &particle.color, sizeof(int));
        std::memcpy(&data[20], &particle.anti_color, sizeof(int));
        std::memcpy(&data[24], &particle.px, sizeof(double));
        std::memcpy(&data[32], &particle.py, sizeof(double));
        std::memcpy(&data[40], &particle.pz, sizeof(double));
        std::memcpy(&data[48], &particle.energy, sizeof(double));
        std::memcpy(&data[56], &particle.mass, sizeof(double));
        std::memcpy(&data[64], &particle.lifetime, sizeof(double));
        std::memcpy(&data[72], &particle.spin, sizeof(double));
    }
    std::array<char, 6 * sizeof(int) + 7 * sizeof(double)> data;
};

struct PackedLHEEvent {
    static constexpr std::array<FieldDescr, 5> field_descr = {{
        {"process_id", "<i4"},
        {"weight", "<f8"},
        {"scale", "<f8"},
        {"alpha_qed", "<f8"},
        {"alpha_qcd", "<f8"},
    }};

    PackedLHEEvent(const LHEEvent& event) {
        std::memcpy(&data[0], &event.process_id, sizeof(int));
        std::memcpy(&data[4], &event.weight, sizeof(double));
        std::memcpy(&data[12], &event.scale, sizeof(double));
        std::memcpy(&data[20], &event.alpha_qed, sizeof(double));
        std::memcpy(&data[28], &event.alpha_qcd, sizeof(double));
    }
    std::array<char, 1 * sizeof(int) + 4 * sizeof(double)> data;
};

template<typename E, typename P>
class EventBuffer {
public:
    EventBuffer(std::size_t event_count, std::size_t particle_count) :
        _event_count(event_count),
        _particle_count(particle_count),
        _data(event_count * (sizeof(E) + particle_count * sizeof(P)))
    {}
    char* data() { return _data.data(); }
    const char* data() const { return _data.data(); }
    std::size_t size() const { return _data.size(); }
    std::size_t event_count() const { return _event_count; }
    std::size_t particle_count() const { return _particle_count; }

    std::size_t event_size() const {
        return sizeof(E) + _particle_count * sizeof(P);
    }

    std::size_t event_offset(std::size_t event_index) const {
        return event_index * event_size();
    }

    std::size_t particle_offset(std::size_t event_index, std::size_t particle_index) const {
        return event_offset(event_index)
            + sizeof(E) + particle_index * sizeof(P);
    }

    P particle(std::size_t event_index, std::size_t particle_index) const {
        P record;
        std::size_t offset = particle_offset(particle_index, event_index);
        std::memcpy(&record, &_data.data()[offset], sizeof(P));
        return record;
    }

    void set_particle(std::size_t event_index, std::size_t particle_index, P record) {
        std::size_t offset = particle_offset(particle_index, event_index);
        std::memcpy(&_data.data()[offset], &record, sizeof(P));
    }

    E event(std::size_t event_index) const {
        E record;
        std::size_t offset = event_offset(event_index);
        std::memcpy(&record, &_data.data()[offset], sizeof(E));
        return record;
    }

    void set_event(std::size_t event_index, E record) {
        std::size_t offset = event_offset(event_index);
        std::memcpy(&_data.data()[offset], &record, sizeof(E));
    }

    void resize(std::size_t event_count) {
        _data.resize(event_count * event_size());
        _event_count = event_count;
    }

    void copy_and_pad(const EventBuffer<E,P>& buffer) {
        if (buffer.particle_count() > particle_count()) {
            throw std::runtime_error("Given buffer contains too many particles");
        }
        resize(buffer.event_count());
        for (std::size_t i = 0; i < event_count(); ++i) {
            std::memcpy(
                &data()[event_offset(i)],
                &buffer.data()[buffer.event_offset(i)],
                buffer.event_size()
            );
            for (std::size_t j = buffer.particle_count(); j < particle_count(); ++j) {
                set_particle(i, j, P{});
            }
        }
    }
private:
    std::size_t _event_count;
    std::size_t _particle_count;
    std::vector<char> _data;
};

class EventFile {
public:
    enum Mode { create, append, load };

    struct DataDescr {
        std::span<FieldDescr> event_fields;
        std::span<FieldDescr> particle_fields;
        std::size_t event_size;
        std::size_t particle_size;
    };

    template<typename E, typename P>
    static DataDescr descr() {
        return {
            .event_fields = {E::field_descr.begin(), E::field_descr.end()},
            .particle_fields = {P::field_descr.begin(), P::field_descr.end()},
            .event_size = sizeof(E),
            .particle_size = sizeof(P),
        };
    }

    EventFile(
        const std::string& file_name,
        DataDescr descr,
        std::size_t particle_count = 0,
        Mode mode = create,
        bool delete_on_close = false
    );

    EventFile(EventFile&& other) noexcept = default;
    EventFile& operator=(EventFile&& other) noexcept = default;
    void seek(std::size_t index);
    void clear();
    std::size_t particle_count() const { return _particle_count; }
    std::size_t event_count() const { return _event_count; }
    ~EventFile();

    template<typename E, typename P>
    void write(const EventBuffer<E,P>& event) {
        if (_mode == EventFile::load) {
            throw std::runtime_error("Event file opened in read mode.");
        }
        if (event.particles().size() != _particle_count) {
            throw std::invalid_argument("Wrong number of particles");
        }
        _file_stream.write(event.data(), event.size());
        ++_current_event;
        if (_current_event > _event_count) {
            _event_count = _current_event;
        }
    }

    template<typename E, typename P>
    bool read(EventBuffer<E,P>& buffer, std::size_t count) {
        if (_current_event == _event_count) return false;
        count = std::min(count, _event_count - _current_event);
        buffer.resize(count);
        if (buffer.particle_count() == _particle_count) {
            _file_stream.read(buffer.data(), buffer.size());
        } else if (buffer.particle_count() > _particle_count) {
            EventBuffer<E,P> tmp_buffer(count, _particle_count);
            _file_stream.read(tmp_buffer.data(), tmp_buffer.size());
            buffer.copy_and_pad(tmp_buffer);
        } else {
            throw std::invalid_argument("Wrong number of particles");
        }
        //if (_file_stream.fail()) return false;
        ++_current_event;
        return true;
    }

private:
    std::string _file_name;
    std::size_t _event_count;
    std::size_t _current_event;
    std::size_t _capacity;
    std::size_t _particle_count;
    std::size_t _shape_pos;
    std::fstream _file_stream;
    std::size_t _header_size;
    std::size_t _event_size;
    Mode _mode;
    bool _delete_on_close;
};

}

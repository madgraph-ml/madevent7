#include "madevent/driver/io.h"

#include <algorithm>
#include <print>
#include <cctype>
#include <filesystem>

#include <nlohmann/json.hpp>

using namespace madevent;
using json = nlohmann::json;

namespace {

std::string dtype_to_str(DataType dtype) {
    switch (dtype) {
    case DataType::dt_float:
        return "<f8";
        break;
    case DataType::dt_int:
        return "<i8";
        break;
    case DataType::dt_bool:
        return "|b1";
        break;
    default:
        throw std::invalid_argument("Unsupported data type");
    }
}

DataType str_to_dtype(std::string dtype) {
    if (dtype == "<f8") {
        return DataType::dt_float;
    } else if (dtype == "<i8") {
        return DataType::dt_int;
    } else if (dtype == "|b1") {
        return DataType::dt_bool;
    } else {
        throw std::invalid_argument("Unsupported data type");
    }
}

json parse_header(std::fstream& file_stream) {
    std::string header;
    auto header_end = header.end();
    header.erase(
        std::remove_if(
            header.begin(), header.end(), [](char x) { return std::isspace(x); }
        ),
        header.end()
    );
    std::replace(header.begin(), header.end(), '(', '[');
    std::replace(header.begin(), header.end(), ')', ']');
    std::size_t pos = 0;
    while ((pos = header.find(",}", pos)) != std::string::npos) {
        header.erase(pos, 1);
    }
    pos = 0;
    while ((pos = header.find(",]", pos)) != std::string::npos) {
        header.erase(pos, 1);
    }
    pos = 0;
    while ((pos = header.find("False", pos)) != std::string::npos) {
        header[pos] = 'f';
    }
    pos = 0;
    while ((pos = header.find("True", pos)) != std::string::npos) {
        header[pos] = 't';
    }
    return json::parse(header);
}

std::tuple<std::size_t, std::size_t, std::size_t> read_event_header(std::fstream& file_stream) {
    json header = parse_header(file_stream);
    if (!header.is_object()) {
        throw std::runtime_error("Invalid header");
    }
    json descr = header.at("descr");
    json fortran_order = header.at("fortran_order");
    json header_shape = header.at("shape");
    if (
        !descr.is_array() ||
        (descr.size() - 1) % 5 != 0 ||
        !fortran_order.is_boolean() ||
        fortran_order.get<bool>() ||
        header_shape.is_array() ||
        header_shape.size() != 1
    ) {
        throw std::runtime_error("Invalid header for event file");
    }
    auto particle_count = (descr.size() - 1) / 5;
    auto event_count = (descr.size() - 1) / 5;
    json weight_descr = descr.at(0);
    if (
        !weight_descr.is_array() ||
        weight_descr.size() != 2 ||
        weight_descr.at(0) != "w" ||
        weight_descr.at(1) != "<i8"
    ) {
        throw std::runtime_error("Invalid header for event file");
    }
    auto descr_iter = descr.begin() + 1;
    const std::tuple<std::string,std::string> particle_descr[5] = {
        {"pid", "<i8"}, {"e", "<f8"}, {"px", "<f8"}, {"py", "<f8"}, {"pz", "<f8"}
    };
    for (std::size_t i = 1; i <= particle_count; ++i) {
        for (auto& [name_str, desc_str] : particle_descr) {
            auto& descr_item = *(descr_iter++);
            if (
                !descr_item.is_array() ||
                descr_item.size() != 2 ||
                descr_item.at(0) != std::format("{}{}", name_str, i) ||
                descr_item.at(1) != desc_str
            ) {
                throw std::runtime_error("Invalid header for event file");
            }
        }
    }
    auto header_size = 0;
    return {header_size, particle_count, event_count};
}

std::tuple<std::size_t, std::size_t> write_event_header(
    std::fstream& file_stream, std::size_t particle_count, std::size_t header_size = 0
) {
    using namespace std::string_literals;
    file_stream << "\x93NUMPY\x01\x00\x00\x00{'descr':[('w','<f8'),"s;
    for (std::size_t i = 1; i <= particle_count; ++i) {
        std::print(
            file_stream,
            "('pid{}','<i8'),('e{}','<f8'),('px{}','<f8'),('py{}','<f8'),('pz{}','<f8'),",
            i, i, i, i, i
        );
    }
    file_stream << "],'fortran_order':False,'shape':(";
    std::size_t shape_pos = file_stream.tellp();
    if (header_size == 0) {
        header_size = (shape_pos + 100) / 64 * 64;
    }
    for (int i = shape_pos; i < header_size - 1; ++i) {
        file_stream.put(' ');
    }
    file_stream.put('\n');
    file_stream.seekp(8);
    uint16_t header_size_short = header_size - 10;
    file_stream.write(reinterpret_cast<char*>(&header_size_short), 2);
    file_stream.seekp(header_size);
    return {header_size, shape_pos};
}

}

Tensor madevent::load_tensor(const std::string& file) {
    std::fstream file_stream(file, std::ios::binary);
    json header = parse_header(file_stream);
    if (!header.is_object()) {
        throw std::runtime_error("Invalid header");
    }
    json descr = header.at("descr");
    json fortran_order = header.at("fortran_order");
    json header_shape = header.at("shape");
    if (
        !descr.is_string() ||
        !fortran_order.is_boolean() ||
        !fortran_order.get<bool>() ||
        header_shape.is_array()
    ) {
        throw std::runtime_error("Invalid file header");
    }
    DataType dtype = str_to_dtype(descr);
    SizeVec shape;
    for (auto& size : header_shape) {
        if (!size.is_number_unsigned()) {
            throw std::runtime_error("Invalid file header");
        }
        shape.push_back(size.get<std::size_t>());
    }
    Tensor tensor(dtype, shape);
    file_stream.read(reinterpret_cast<char*>(tensor.data()), tensor.byte_size());
    if (file_stream.fail()) {
        throw std::runtime_error("Failed to read file");
    }
    return tensor;
}

void madevent::save_tensor(const std::string& file, Tensor tensor) {
    Tensor cpu_tensor = tensor.cpu().contiguous();
    std::ofstream file_stream(file, std::ios::binary);
    file_stream
        << "\x93NUMPY\x01\x00\x76\x00{'descr':'"
        << dtype_to_str(tensor.dtype())
        << "','fortran_order':True,'shape':(";
    for (std::size_t size : tensor.shape()) {
        file_stream << size << ",";
    }
    file_stream << ")}";
    for (int i = file_stream.tellp(); i < 127; ++i) {
        file_stream.put(' ');
    }
    file_stream.put('\n');
    file_stream.write(
        reinterpret_cast<const char*>(cpu_tensor.data()), cpu_tensor.byte_size()
    );
}

EventFile::EventFile(const std::string& file_name, std::size_t particle_count, EventFile::Mode mode) :
    _file_name(file_name),
    _event_count(0),
    _current_event(0),
    _capacity(0),
    _particle_count(particle_count),
    _mode(mode)
{
    auto file_mode = std::ios::binary | std::ios::in;
    if (mode == EventFile::create) {
        file_mode |= std::ios::out | std::ios::trunc;
    } else if (mode == EventFile::append) {
        file_mode |= std::ios::out;
    }
    _file_stream.open(file_name, file_mode);
    if (!_file_stream) {
        throw std::runtime_error(std::format("Could not open file {}", file_name));
    }
    if (mode == EventFile::create) {
        std::tie(_header_size, _shape_pos) =
            write_event_header(_file_stream, _particle_count);
    } else {
        std::tie(_header_size, _particle_count, _event_count) =
            read_event_header(_file_stream);
        if (mode == EventFile::load) {
            std::tie(_header_size, _shape_pos) = write_event_header(
                _file_stream, _particle_count, _header_size
            );
        }
        _capacity = _event_count;
    }
}

void EventFile::write(const EventBuffer& event) {
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

bool EventFile::read(EventBuffer& event) {
    if (_current_event == _event_count) return false;
    _file_stream.read(event.data(), event.size());
    //if (_file_stream.fail()) return false;
    ++_current_event;
    return true;
}

void EventFile::seek(std::size_t index) {
    _current_event = index;
    _file_stream.seekp(_header_size + index * EventBuffer::size(_particle_count));
}

std::size_t EventFile::unweight(double max_weight, std::function<double()> random_generator) {
    if (_mode == EventFile::load) {
        throw std::runtime_error("Event file opened in read mode.");
    }

    EventBuffer buffer(_particle_count);
    std::size_t accept_count = 0;
    for (std::size_t i = 0; i < _event_count; ++i) {
        seek(i);
        read(buffer);
        double& weight = buffer.event().weight;
        if (weight / max_weight < random_generator()) {
            weight = 0;
        } else {
            weight = std::max(weight, max_weight);
            ++accept_count;
        }
        seek(i);
        write(buffer);
    }

    return accept_count;
}

void EventFile::clear() {
    if (_mode == EventFile::load) {
        throw std::runtime_error("Event file opened in read mode.");
    }

    _capacity = _current_event;
    seek(0);
    _event_count = 0;
}

EventFile::~EventFile() {
    if (!_file_stream.is_open() || _mode == EventFile::load) return;
    _file_stream.seekp(_shape_pos);
    _file_stream << _event_count << ",)}";
    if (_event_count < _capacity) {
        _file_stream.close();
        std::filesystem::resize_file(
            _file_name, _header_size + _event_count * EventBuffer::size(_particle_count)
        );
    }
}

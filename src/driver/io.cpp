#include "madevent/driver/io.h"

#include <algorithm>
#include <print>
#include <cctype>

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

json parse_header(std::ifstream& file_stream) {
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

}

Tensor madevent::load_tensor(const std::string& file) {
    std::ifstream file_stream(file, std::ios::binary);
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

EventReader::EventReader(const std::string& file) :
    file_stream(file, std::ios::binary)
{
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
    particle_count = (descr.size() - 1) / 5;
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
}

bool EventReader::read(EventRecord& event) {
    return !file_stream.fail();
}

EventWriter::EventWriter(const std::string& file, std::size_t particle_count) :
    event_count(0),
    particle_count(particle_count),
    file_stream(file, std::ios::binary)
{
    file_stream << "\x93NUMPY\x01\x00\x76\x00{'descr':[('w','<f8'),";
    for (std::size_t i = 1; i <= particle_count; ++i) {
        std::print(
            file_stream,
            "('pid{}','<i8'),('e{}','<f8'),('px{}','<f8'),('py{}','<f8'),('pz{}','<f8'),",
            i, i, i, i, i
        );
    }
    file_stream << "],'fortran_order':False,'shape':(";
    shape_pos = file_stream.tellp();
    std::size_t header_size = (shape_pos + 100) / 64 * 64;
    for (int i = shape_pos; i < header_size - 1; ++i) {
        file_stream.put(' ');
    }
    file_stream.put('\n');
}

void EventWriter::write(const EventRecord& event) {
    if (event.particles.size() != particle_count) {
        throw std::invalid_argument("Wrong number of particles");
    }
    file_stream.write(
        reinterpret_cast<const char*>(&event.weight),
        sizeof(long long)
    );
    file_stream.write(
        reinterpret_cast<const char*>(event.particles.data()),
        particle_count * sizeof(ParticleRecord)
    );
    ++event_count;
}

EventWriter::~EventWriter() {
    file_stream.seekp(shape_pos);
    file_stream << event_count << ",)}";
}

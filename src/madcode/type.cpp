#include "madevent/madcode/type.h"
#include "madevent/util.h"

using namespace madevent;

std::size_t BatchSize::UnnamedBody::counter = 0;
const BatchSize BatchSize::zero = BatchSize(BatchSize::Compound{});
const BatchSize BatchSize::one = BatchSize(BatchSize::One{});

BatchSize BatchSize::add(const BatchSize& other, int factor) const {
    BatchSize::Compound compound;
    std::visit(Overloaded{
        [&](Compound val) { compound = val; },
        [&](auto val) { compound[val] = 1; }
    }, value);
    std::visit(Overloaded{
        [&](Compound val) {
            for (auto& [sub_key, sub_val] : val) {
                compound[sub_key] += factor * sub_val;
            }
        },
        [&](auto val) { compound[val] += factor; }
    }, other.value);
    for (auto key_val = compound.begin(); key_val != compound.end();) {
        if (key_val->second == 0) {
            compound.erase(key_val++);
        } else {
            ++key_val;
        }
    }
    if (compound.size() == 1) {
        auto& [key, val] = *compound.begin();
        if (val == 1) {
            return std::visit([](auto& k) { return BatchSize(k); }, key);
        }
    }
    return compound;
}

std::ostream& madevent::operator<<(std::ostream& out, const DataType& dtype) {
    switch (dtype) {
    case DataType::dt_bool:
        out << "bool";
        break;
    case DataType::dt_float:
        out << "float";
        break;
    case DataType::dt_int:
        out << "int";
        break;
    case DataType::batch_sizes:
        out << "batch_sizes";
        break;
    }
    return out;
}

std::ostream& madevent::operator<<(std::ostream& out, const BatchSize& batch_size) {
    std::visit(Overloaded {
        [&](BatchSize::Named value) { out << value; },
        [&](BatchSize::Unnamed value) { out << "$" << value->id; },
        [&](BatchSize::One value) { out << "1"; },
        [&](BatchSize::Compound value) {
            if (value.size() == 0) {
                out << "0";
                return;
            }
            bool first = true;
            for (auto& [key, count] : value) {
                if (count == 1) {
                    if (!first) out << "+";
                } else if (count == -1) {
                    out << "-";
                } else {
                    if (count > 1 && !first) out << "+";
                    out << count << "*";
                }
                std::visit([&](auto& k) { out << BatchSize(k); }, key);
                first = false;
            }
        }
    }, batch_size.value);
    return out;
}

std::ostream& madevent::operator<<(std::ostream& out, const Type& type) {
    if (type.dtype == DataType::batch_sizes) {
        out << "{";
        bool first = true;
        for (auto& batch_size : type.batch_size_list) {
            if (first) {
                first = false;
            } else {
                out << ", ";
            }
            out << batch_size;
        }
        out << "}";
    } else {
        out << type.dtype << "[" << type.batch_size;
        for (int size : type.shape) {
            out << ", " << size;
        }
        out << "]";
    }
    return out;
}

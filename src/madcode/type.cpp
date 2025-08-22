#include "madevent/madcode/type.h"
#include "madevent/util.h"

using namespace madevent;
using json = nlohmann::json;

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

Type madevent::multichannel_batch_size(int count) {
    std::vector<BatchSize> batch_sizes;
    BatchSize remaining = batch_size;
    for (std::size_t i = 0; i < count - 1; ++i) {
        BatchSize batch_size_i(std::format("channel_size_{}", i));
        batch_sizes.push_back(batch_size_i);
        remaining = remaining - batch_size_i;
    }
    batch_sizes.push_back(remaining);
    return batch_sizes;
}

void madevent::to_json(json& j, const BatchSize& batch_size) {
    std::visit(Overloaded {
        [&](BatchSize::Named value) { j = value; },
        [&](BatchSize::Unnamed value) { j = nullptr; },
        [&](BatchSize::One value) { j = 1; },
        [&](BatchSize::Compound value) {
            j = json(json::value_t::array);
            for (auto& [key, factor] : value) {
                j.push_back(json{
                    {
                        "batch_size",
                        std::visit([&](auto& k) { return json{BatchSize(k)}; }, key)
                    },
                    {"factor", factor}
                });
            }
        }
    }, batch_size.value);
}

void madevent::to_json(json& j, const Value& value) {
    std::visit(Overloaded{
        [&](auto val) {
            j = json{
                {"dtype", value.type.dtype},
                {"shape", json(json::value_t::array)},
                {"data", val}
            };
        },
        [&](TensorValue val) {
            j = json{
                {"dtype", value.type.dtype},
                {"shape", std::get<0>(val)},
                {"data", std::visit(
                    [](auto data) { return json{data}; }, std::get<1>(val)
                )}
            };
        },
        [&](std::monostate val) {
            j = value.local_index;
        }
    }, value.literal_value);
}

void madevent::to_json(json& j, const DataType& dtype) {
    switch (dtype) {
        case DataType::dt_int: j = "int"; break;
        case DataType::dt_float: j = "float"; break;
        case DataType::batch_sizes: j = "batch_sizes"; break;
    }
}

void madevent::from_json(const json& j, BatchSize& batch_size) {
    if (j.is_string()) {
        batch_size = j.get<std::string>();
    } else if (j.is_number_integer() && j.get<int>() == 1) {
        batch_size = BatchSize::one;
    } else if (j.is_array()) {
        BatchSize::Compound compound;
        for (auto& j_item : j) {
            std::visit(Overloaded {
                [&](auto value) {
                    compound[value] = j_item.at("factor").get<int>();
                },
                [&](BatchSize::Compound value) {
                    throw std::invalid_argument("invalid batch size");
                }
            }, j_item.at("batch_size").get<BatchSize>().value);
        }
        batch_size = compound;
    } else {
        throw std::invalid_argument("invalid batch size");
    }
}

void madevent::from_json(const json& j, Value& value) {
    auto dtype = j.at("dtype").get<DataType>();
    auto shape = j.at("shape").get<std::vector<int>>();
    auto j_data = j.at("data");
    switch(dtype) {
    case DataType::dt_int:
        if (shape.size() == 0) {
            value = j_data.get<me_int_t>();
        } else {
            value = Value(j_data.get<std::vector<me_int_t>>(), shape);
        }
    case DataType::dt_float:
        if (shape.size() == 0) {
            value = j_data.get<double>();
        } else {
            value = Value(j_data.get<std::vector<double>>(), shape);
        }
        break;
    case DataType::batch_sizes:
        throw std::invalid_argument("invalid data type");
    }
}

void madevent::from_json(const json& j, DataType& dtype) {
    auto str = j.get<std::string>();
    if (str == "int") {
        dtype = DataType::dt_int;
    } else if (str == "float") {
        dtype = DataType::dt_float;
    } else if (str == "batch_sizes") {
        dtype = DataType::batch_sizes;
    } else {
        throw std::invalid_argument("invalid data type");
    }
}


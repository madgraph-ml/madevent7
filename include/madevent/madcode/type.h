#pragma once

#include <string>
#include <variant>
#include <vector>
#include <unordered_map>
#include <iostream>

namespace madevent {

enum class DataType {
    dt_bool,
    dt_int,
    dt_float,
    batch_sizes
};

class BatchSize {
public:
    using Named = std::string;
    class UnnamedBody {
    public:
        UnnamedBody() : id(counter++) {}
        friend std::ostream& operator<<(std::ostream& out, const BatchSize& batch_size);
        bool operator==(const UnnamedBody& other) const { return id == other.id; }
        bool operator!=(const UnnamedBody& other) const { return id != other.id; }
    private:
        static std::size_t counter;
        std::size_t id;
    };
    using Unnamed = std::shared_ptr<UnnamedBody>;
    using One = std::monostate;
    using Compound = std::unordered_map<std::variant<Named, Unnamed, One>, int>;

    static const BatchSize zero;
    static const BatchSize one;

    BatchSize(std::string name) : value(name) {}
    BatchSize(One value) : value(value) {}
    BatchSize() : value(std::make_shared<UnnamedBody>()) {}
    BatchSize operator+(const BatchSize& other) const { return add(other, 1); }
    BatchSize operator-(const BatchSize& other) const { return add(other, -1); }
    bool operator==(const BatchSize& other) const { return value == other.value; }
    bool operator!=(const BatchSize& other) const { return value != other.value; }

    friend std::ostream& operator<<(std::ostream& out, const BatchSize& batch_size);

private:
    BatchSize(Compound value) : value(value) {}
    BatchSize(Unnamed value) : value(value) {}
    BatchSize add(const BatchSize& other, int factor) const;

    std::variant<Named, Unnamed, One, Compound> value;
};

struct Type {
    DataType dtype;
    BatchSize batch_size;
    std::vector<int> shape;
    std::vector<BatchSize> batch_size_list;

    Type(DataType _dtype, BatchSize _batch_size, std::vector<int> _shape) :
        dtype(_dtype), batch_size(_batch_size), shape(_shape) {}
    Type(std::vector<BatchSize> _batch_size_list) :
        dtype(DataType::batch_sizes),
        batch_size(BatchSize::one),
        batch_size_list(_batch_size_list)
    {}
};

std::ostream& operator<<(std::ostream& out, const DataType& dtype);
std::ostream& operator<<(std::ostream& out, const Type& type);

inline bool operator==(const Type& lhs, const Type& rhs) {
    return lhs.dtype == rhs.dtype && lhs.batch_size == rhs.batch_size && lhs.shape == rhs.shape;
}

inline bool operator!=(const Type& lhs, const Type& rhs) {
    return lhs.dtype != rhs.dtype || lhs.batch_size != rhs.batch_size || lhs.shape != rhs.shape;
}

using TypeList = std::vector<Type>;

const Type single_float{DataType::dt_float, BatchSize::One{}, {}};
const Type single_int{DataType::dt_int, BatchSize::One{}, {}};
const Type single_bool{DataType::dt_bool, BatchSize::One{}, {}};
inline Type single_int_array(int count) {
    return {DataType::dt_int, BatchSize::one, {count}};
}

const BatchSize batch_size = BatchSize("batch_size");
const Type batch_float{DataType::dt_float, batch_size, {}};
const Type batch_int{DataType::dt_int, batch_size, {}};
const Type batch_bool{DataType::dt_bool, batch_size, {}};
const Type batch_four_vec{DataType::dt_float, batch_size, {4}};
inline Type batch_float_array(int count) {
    return {DataType::dt_float, batch_size, {count}};
}
inline Type batch_four_vec_array(int count) {
    return {DataType::dt_float, batch_size, {count, 4}};
}


using TensorValue = std::tuple<
    std::vector<int>, std::variant<std::vector<bool>, std::vector<long long>, std::vector<double>>
>;

using LiteralValue = std::variant<bool, long long, double, TensorValue, std::monostate>;

struct Value {
    Type type;
    LiteralValue literal_value;
    int local_index = -1;

    Value() : type(single_float), literal_value(std::monostate{}) {}

    Value(bool value) : type(single_bool), literal_value(value) {}
    Value(long long value) : type(single_int), literal_value(value) {}
    Value(double value) : type(single_float), literal_value(value) {}

    template<typename T>
    Value(const std::vector<T>& values, const std::vector<int>& shape = {}) :
        type{
            std::is_same_v<T, bool> ? DataType::dt_bool :
            std::is_same_v<T, long long> ? DataType::dt_int : DataType::dt_float,
            BatchSize::one,
            shape.size() == 0 ? std::vector<int>{static_cast<int>(values.size())} : shape
        },
        literal_value(TensorValue(type.shape, values))
    {
        std::size_t prod = 1;
        for (auto size : type.shape) {
            prod *= size;
        }
        if (prod != values.size()) {
            throw std::invalid_argument("size of value vector not compatible with given shape");
        }
    }

    Value(Type _type, int _local_index)
        : type(_type), literal_value(std::monostate{}), local_index(_local_index) {}
    Value(Type _type, LiteralValue _literal_value, int _local_index = -1)
        : type(_type), literal_value(_literal_value), local_index(_local_index) {}
};

using ValueList = std::vector<Value>;

}

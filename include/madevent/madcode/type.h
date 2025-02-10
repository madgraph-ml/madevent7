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
    std::optional<BatchSize> broadcast(const BatchSize& other) const;

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

}

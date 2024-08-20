#pragma once

#include <vector>
#include <string>
#include <tuple>
#include <variant>
#include <initializer_list>
#include <memory>
#include <unordered_map>

namespace madevent {

enum DataType {
    DT_BOOL,
    DT_INT,
    DT_FLOAT
};

struct Type {
    DataType dtype;
    std::vector<int> shape;
};

using TypeList = std::vector<Type>;

inline bool operator==(const Type& lhs, const Type& rhs) {
    return lhs.dtype == rhs.dtype && lhs.shape == rhs.shape;
}

inline bool operator!=(const Type& lhs, const Type& rhs) {
    return lhs.dtype != rhs.dtype || lhs.shape != rhs.shape;
}

const Type scalar{DT_FLOAT, {}};
const Type scalar_int{DT_INT, {}};
const Type scalar_bool{DT_BOOL, {}};
const Type four_vector{DT_FLOAT, {4}};
inline Type four_vector_array(int count) { return {DT_FLOAT, {count, 4}}; }

class Instruction {
public:
    std::string name;
    int opcode;

    Instruction(std::string _name, int _opcode) : name(_name), opcode(_opcode) {}
    virtual ~Instruction() = default;
    virtual const TypeList signature(const TypeList& args) const = 0;
};

class SimpleInstruction : public Instruction {
public:
    using DynShape = std::vector<std::variant<int, std::string>>;
    using SigType = std::tuple<DataType, DynShape>;

    SimpleInstruction(
        std::string _name,
        int _opcode,
        std::initializer_list<SigType> _inputs,
        std::initializer_list<SigType> _outputs
    ) : Instruction(_name, _opcode), inputs(_inputs), outputs(_outputs) {}

    const TypeList signature(const TypeList& args) const override;

private:
    const std::vector<SigType> inputs;
    const std::vector<SigType> outputs;
};

class PrintInstruction : public Instruction {
public:
    PrintInstruction(int _opcode) : Instruction("print", _opcode) {}
    const TypeList signature(const TypeList& args) const override {
        return {};
    }
};

using InstructionPtr = std::unique_ptr<Instruction>;
const std::unordered_map<std::string, InstructionPtr> build_instruction_set();
const std::unordered_map<std::string, InstructionPtr> instruction_set = build_instruction_set();

}

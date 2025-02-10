#pragma once

#include <vector>
#include <string>
#include <tuple>
#include <variant>
#include <initializer_list>
#include <memory>
#include <unordered_map>

#include "type.h"

namespace madevent {

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
    return {DataType::dt_bool, batch_size, {count}};
}
inline Type batch_four_vec_array(int count) {
    return {DataType::dt_bool, batch_size, {count, 4}};
}

enum Opcode {
#include "opcode_mixin.h"
};

class Instruction {
public:
    std::string name;
    int opcode;

    Instruction(std::string _name, int _opcode) : name(_name), opcode(_opcode) {}
    virtual ~Instruction() = default;
    virtual TypeList signature(const TypeList& args) const = 0;
};

class SimpleInstruction : public Instruction {
public:
    using DynShape = std::vector<std::variant<int, std::string>>;
    using SigType = std::tuple<DataType, bool, DynShape>;

    SimpleInstruction(
        std::string _name,
        int _opcode,
        std::initializer_list<SigType> _inputs,
        std::initializer_list<SigType> _outputs
    ) : Instruction(_name, _opcode), inputs(_inputs), outputs(_outputs) {}

    TypeList signature(const TypeList& args) const override;

private:
    const std::vector<SigType> inputs;
    const std::vector<SigType> outputs;
};

class PrintInstruction : public Instruction {
public:
    PrintInstruction(int _opcode) : Instruction("print", _opcode) {}
    TypeList signature(const TypeList& args) const override {
        return {};
    }
};

class StackInstruction : public Instruction {
public:
    StackInstruction(int _opcode) : Instruction("stack", _opcode) {}
    TypeList signature(const TypeList& args) const override;
};

class UnstackInstruction : public Instruction {
public:
    UnstackInstruction(int _opcode) : Instruction("unstack", _opcode) {}
    TypeList signature(const TypeList& args) const override;
};

class BatchCatInstruction : public Instruction {
public:
    BatchCatInstruction(int _opcode) : Instruction("batch_cat", _opcode) {}
    TypeList signature(const TypeList& args) const override;
};

class BatchSplitInstruction : public Instruction {
public:
    BatchSplitInstruction(int _opcode) : Instruction("batch_split", _opcode) {}
    TypeList signature(const TypeList& args) const override;
};

using InstructionOwner = std::unique_ptr<const Instruction>;
using InstructionPtr = Instruction const*;
const std::unordered_map<std::string, InstructionOwner> build_instruction_set();
const std::unordered_map<std::string, InstructionOwner> instruction_set = build_instruction_set();

}

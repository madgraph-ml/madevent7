#pragma once

#include <map>
#include <vector>
#include <string>
#include <tuple>
#include <variant>
#include <initializer_list>
#include <memory>
#include <unordered_map>

#include "type.h"

namespace madevent {

enum Opcode {
#include "opcode_mixin.h"
};

class Instruction {
public:
    std::string name;
    int opcode;

    Instruction(std::string _name, int _opcode) : name(_name), opcode(_opcode) {}
    virtual ~Instruction() = default;
    virtual TypeList signature(const ValueList& args) const = 0;
};

class ShapeExpr {
public:
    ShapeExpr(const char* expr);
    bool check_and_update(std::map<char, int>& variables, int value) const;
    std::optional<int> evaluate(const std::map<char, int>& variables) const;
    char first_var_name() const {
        return std::get<0>(terms.at(0));
    }
private:
    std::vector<std::tuple<char, int>> terms;
};

class SimpleInstruction : public Instruction {
public:
    using DynShape = std::vector<std::variant<int, ShapeExpr, std::monostate>>;
    using SigType = std::tuple<DataType, bool, DynShape, bool>;

    SimpleInstruction(
        std::string _name,
        int _opcode,
        std::initializer_list<SigType> _inputs,
        std::initializer_list<SigType> _outputs
    ) : Instruction(_name, _opcode), inputs(_inputs), outputs(_outputs) {}

    TypeList signature(const ValueList& args) const override;

private:
    const std::vector<SigType> inputs;
    const std::vector<SigType> outputs;
};

class StackInstruction : public Instruction {
public:
    StackInstruction(int _opcode) : Instruction("stack", _opcode) {}
    TypeList signature(const ValueList& args) const override;
};

class UnstackInstruction : public Instruction {
public:
    UnstackInstruction(int _opcode) : Instruction("unstack", _opcode) {}
    TypeList signature(const ValueList& args) const override;
};

class BatchCatInstruction : public Instruction {
public:
    BatchCatInstruction(int _opcode) : Instruction("batch_cat", _opcode) {}
    TypeList signature(const ValueList& args) const override;
};

class BatchSplitInstruction : public Instruction {
public:
    BatchSplitInstruction(int _opcode) : Instruction("batch_split", _opcode) {}
    TypeList signature(const ValueList& args) const override;
};

class RqsActivationInstruction : public Instruction {
public:
    RqsActivationInstruction(int _opcode) : Instruction("rqs_activation", _opcode) {}
    TypeList signature(const ValueList& args) const override;
};

using InstructionOwner = std::unique_ptr<const Instruction>;
using InstructionPtr = Instruction const*;
const std::unordered_map<std::string, InstructionOwner> build_instruction_set();
const std::unordered_map<std::string, InstructionOwner> instruction_set = build_instruction_set();

}

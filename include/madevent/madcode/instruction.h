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
    Instruction(const std::string& name, int opcode) : _name(name), _opcode(opcode) {}
    virtual ~Instruction() = default;
    virtual TypeVec signature(const ValueVec& args) const = 0;
    const std::string& name() const { return _name; }
    int opcode() const { return _opcode; }

protected:
    void check_arg_count(const ValueVec& args, std::size_t count) const;

private:
    std::string _name;
    int _opcode;
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

    TypeVec signature(const ValueVec& args) const override;

private:
    const std::vector<SigType> inputs;
    const std::vector<SigType> outputs;
};

class StackInstruction : public Instruction {
public:
    StackInstruction(int _opcode) : Instruction("stack", _opcode) {}
    TypeVec signature(const ValueVec& args) const override;
};

class UnstackInstruction : public Instruction {
public:
    UnstackInstruction(int _opcode) : Instruction("unstack", _opcode) {}
    TypeVec signature(const ValueVec& args) const override;
};

class BatchCatInstruction : public Instruction {
public:
    BatchCatInstruction(int _opcode) : Instruction("batch_cat", _opcode) {}
    TypeVec signature(const ValueVec& args) const override;
};

class BatchSplitInstruction : public Instruction {
public:
    BatchSplitInstruction(int _opcode) : Instruction("batch_split", _opcode) {}
    TypeVec signature(const ValueVec& args) const override;
};

class RqsActivationInstruction : public Instruction {
public:
    RqsActivationInstruction(int _opcode) : Instruction("rqs_activation", _opcode) {}
    TypeVec signature(const ValueVec& args) const override;
};

class NonzeroInstruction : public Instruction {
public:
    NonzeroInstruction(int _opcode) : Instruction("nonzero", _opcode) {}
    TypeVec signature(const ValueVec& args) const override;
};

class BatchGatherInstruction : public Instruction {
public:
    BatchGatherInstruction(int _opcode) : Instruction("batch_gather", _opcode) {}
    TypeVec signature(const ValueVec& args) const override;
};

class ScatterInstruction : public Instruction {
public:
    ScatterInstruction(int _opcode) : Instruction("scatter", _opcode) {}
    TypeVec signature(const ValueVec& args) const override;
};

class RandomInstruction : public Instruction {
public:
    RandomInstruction(int _opcode) : Instruction("random", _opcode) {}
    TypeVec signature(const ValueVec& args) const override;
};

class UnweightInstruction : public Instruction {
public:
    UnweightInstruction(int _opcode) : Instruction("unweight", _opcode) {}
    TypeVec signature(const ValueVec& args) const override;
};

using InstructionOwner = std::unique_ptr<const Instruction>;
using InstructionPtr = Instruction const*;
const std::unordered_map<std::string, InstructionOwner> build_instruction_set();
const std::unordered_map<std::string, InstructionOwner> instruction_set = build_instruction_set();

}

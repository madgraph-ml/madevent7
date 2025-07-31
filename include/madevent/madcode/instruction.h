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

namespace opcodes {
enum Opcode {
#include "opcode_mixin.h"
};
}

class Instruction {
public:
    Instruction(const std::string& name, int opcode, bool differentiable) :
        _name(name), _opcode(opcode), _differentiable(differentiable) {}
    virtual ~Instruction() = default;
    virtual TypeVec signature(const ValueVec& args) const = 0;
    const std::string& name() const { return _name; }
    int opcode() const { return _opcode; }
    bool differentiable() const { return  _differentiable; }

protected:
    void check_arg_count(const ValueVec& args, std::size_t count) const;

private:
    std::string _name;
    int _opcode;
    bool _differentiable;
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
        std::string name,
        int opcode,
        bool differentiable,
        std::initializer_list<SigType> _inputs,
        std::initializer_list<SigType> _outputs
    ) : Instruction(name, opcode, differentiable), inputs(_inputs), outputs(_outputs) {}

    TypeVec signature(const ValueVec& args) const override;

private:
    const std::vector<SigType> inputs;
    const std::vector<SigType> outputs;
};

class StackInstruction : public Instruction {
public:
    StackInstruction(int opcode, bool differentiable) :
        Instruction("stack", opcode, differentiable) {}
    TypeVec signature(const ValueVec& args) const override;
};

class UnstackInstruction : public Instruction {
public:
    UnstackInstruction(int opcode, bool differentiable) :
        Instruction("unstack", opcode, differentiable) {}
    TypeVec signature(const ValueVec& args) const override;
};

class UnstackSizesInstruction : public Instruction {
public:
    UnstackSizesInstruction(int opcode, bool differentiable) :
        Instruction("unstack_sizes", opcode, differentiable) {}
    TypeVec signature(const ValueVec& args) const override;
};

class BatchCatInstruction : public Instruction {
public:
    BatchCatInstruction(int opcode, bool differentiable) :
        Instruction("batch_cat", opcode, differentiable) {}
    TypeVec signature(const ValueVec& args) const override;
};

class BatchSplitInstruction : public Instruction {
public:
    BatchSplitInstruction(int opcode, bool differentiable) :
        Instruction("batch_split", opcode, differentiable) {}
    TypeVec signature(const ValueVec& args) const override;
};

class CatInstruction : public Instruction {
public:
    CatInstruction(int opcode, bool differentiable) :
        Instruction("cat", opcode, differentiable) {}
    TypeVec signature(const ValueVec& args) const override;
};

class BatchSizeInstruction : public Instruction {
public:
    BatchSizeInstruction(int opcode, bool differentiable) :
        Instruction("batch_size", opcode, differentiable) {}
    TypeVec signature(const ValueVec& args) const override;
};

class FullInstruction : public Instruction {
public:
    FullInstruction(int opcode, bool differentiable) :
        Instruction("full", opcode, differentiable) {}
    TypeVec signature(const ValueVec& args) const override;
};

class SqueezeInstruction : public Instruction {
public:
    SqueezeInstruction(int opcode, bool differentiable) :
        Instruction("squeeze", opcode, differentiable) {}
    TypeVec signature(const ValueVec& args) const override;
};

class UnsqueezeInstruction : public Instruction {
public:
    UnsqueezeInstruction(int opcode, bool differentiable) :
        Instruction("unsqueeze", opcode, differentiable) {}
    TypeVec signature(const ValueVec& args) const override;
};

class RqsActivationInstruction : public Instruction {
public:
    RqsActivationInstruction(int opcode, bool differentiable) :
        Instruction("rqs_activation", opcode, differentiable) {}
    TypeVec signature(const ValueVec& args) const override;
};

class NonzeroInstruction : public Instruction {
public:
    NonzeroInstruction(int opcode, bool differentiable) :
        Instruction("nonzero", opcode, differentiable) {}
    TypeVec signature(const ValueVec& args) const override;
};

class BatchGatherInstruction : public Instruction {
public:
    BatchGatherInstruction(int opcode, bool differentiable) :
        Instruction("batch_gather", opcode, differentiable) {}
    TypeVec signature(const ValueVec& args) const override;
};

class ScatterInstruction : public Instruction {
public:
    ScatterInstruction(int opcode, bool differentiable) :
        Instruction("scatter", opcode, differentiable) {}
    TypeVec signature(const ValueVec& args) const override;
};

class RandomInstruction : public Instruction {
public:
    RandomInstruction(int opcode, bool differentiable) :
        Instruction("random", opcode, differentiable) {}
    TypeVec signature(const ValueVec& args) const override;
};

class UnweightInstruction : public Instruction {
public:
    UnweightInstruction(int opcode, bool differentiable) :
        Instruction("unweight", opcode, differentiable) {}
    TypeVec signature(const ValueVec& args) const override;
};

using InstructionOwner = std::unique_ptr<const Instruction>;
using InstructionPtr = Instruction const*;
const std::unordered_map<std::string, InstructionOwner> build_instruction_set();
const std::unordered_map<std::string, InstructionOwner> instruction_set = build_instruction_set();

}

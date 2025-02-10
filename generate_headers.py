import yaml

def main():
    with open("src/madcode/instruction_set.yaml") as f:
        data = list(yaml.safe_load_all(f))

    types = data[0]["types"]
    commands = {}
    for sec in data[1:]:
        commands.update({key: value for key, value in sec.items() if key != "title"})
    for i, cmd in enumerate(commands.values()):
        cmd["opcode"] = i
    sections = [
        (sec["title"], [key for key in sec.keys() if key != "title"]) for sec in data[1:]
    ]

    function_builder_mixin(commands)
    instruction_set_python(commands)
    instruction_set_mixin(types, commands)
    cpu_runtime_mixin(commands)


def write_autogen(f):
    f.write(
        "// This file was automatically generated based on instruction_set.yaml\n"
        "// Do not modify its content directly\n\n"
    )


def function_builder_mixin(commands):
    with open("include/madevent/madcode/function_builder_mixin.h", "w") as f:
        write_autogen(f)
        for name, cmd in commands.items():
            if cmd["inputs"] == "any":
                parameters = "ValueList args"
                instruction_call = f"instruction(\"{name}\", args)"
            else:
                parameters = ", ".join(f"Value {arg['name']}" for arg in cmd["inputs"])
                arguments = ", ".join(arg["name"] for arg in cmd["inputs"])
                instruction_call = f"instruction(\"{name}\", {{{arguments}}})"

            if cmd["outputs"] == "any":
                return_type = "ValueList"
                func_body = f"    return {instruction_call};"
            else:
                n_outputs = len(cmd["outputs"])
                if n_outputs == 0:
                    return_type = "void"
                    func_body = f"    {instruction_call};"
                elif n_outputs == 1:
                    return_type = "Value"
                    func_body = f"    return {instruction_call}[0];"
                else:
                    return_type = f"std::array<Value, {n_outputs}>"
                    return_array = ", ".join(f"output_vector[{i}]" for i in range(n_outputs))
                    func_body = (
                        f"    auto output_vector = {instruction_call};\n"
                        f"    return {{{return_array}}};"
                    )

            f.write(f"{return_type} {name}({parameters}) {{\n{func_body}\n}}\n\n");

def instruction_set_python(commands):
    with open("src/python/instruction_set.h", "w") as f:
        write_autogen(f)
        f.write(
            '#pragma once\n\n'
            '#include <pybind11/pybind11.h>\n'
            '#include <pybind11/stl.h>\n'
            '#include "madevent/madcode.h"\n\n'
            'namespace py = pybind11;\n'
            'using madevent::FunctionBuilder;\n\n'
            'namespace {\n\n'
            'void add_instructions(py::class_<FunctionBuilder>& fb) {\n'
        )

        for name, cmd in commands.items():
            f.write(f'    fb.def("{name}", &FunctionBuilder::{name}')
            if cmd["inputs"] == "any":
                f.write(', py::arg("args")')
            else:
                for arg in cmd["inputs"]:
                    f.write(f', py::arg("{arg["name"]}")')
            f.write(');\n')

        f.write('}\n}\n')


def instruction_set_mixin(types, commands):
    with (
        open("src/madcode/instruction_set_mixin.h", "w") as f,
        open("include/madevent/madcode/opcode_mixin.h", "w") as f_op,
    ):
        write_autogen(f)
        f.write("using SigType = SimpleInstruction::SigType;\n")

        for name, sig in types.items():
            dtype = "DataType::dt_" + sig["dtype"]
            single = "true" if sig.get("single", False) else "false"
            shape = ", ".join(
                str(item) if isinstance(item, int) else f"\"{item}\"" for item in sig["shape"]
            )
            f.write(f"const SimpleInstruction::SigType {name} {{{dtype}, {single}, {{{shape}}}}};\n")

        f.write(
            "const auto mi = [](\n"
            "    std::string name,\n"
            "    int opcode,\n"
            "    std::initializer_list<SigType> inputs,\n"
            "    std::initializer_list<SigType> outputs\n"
            ") { return InstructionOwner(new SimpleInstruction(name, opcode, inputs, outputs)); };\n"
            "\n"
            "InstructionOwner instructions[] {\n"
        )
        first = True
        for name, cmd in commands.items():
            opcode = cmd["opcode"]
            if "class" in cmd:
                f.write(f"    InstructionOwner(new {cmd['class']}({opcode})),\n")
            else:
                input_types = ", ".join(arg["type"] for arg in cmd["inputs"])
                output_types = ", ".join(ret["type"] for ret in cmd["outputs"])
                f.write(f"    mi(\"{name}\", {opcode}, {{{input_types}}}, {{{output_types}}}),\n")

            if first:
                first = False
            else:
                f_op.write(",\n")
            f_op.write(f"{name} = {opcode}")
        f.write("};\n")
        f_op.write("\n")


def cpu_runtime_mixin(commands):
    with open("src/backend/cpu/runtime_mixin.h", "w") as f:
        write_autogen(f)

        for name, cmd in commands.items():
            opcode = cmd["opcode"]
            if "class" in cmd:
                func = f"op_{name}"
            else:
                n_inputs = len(cmd["inputs"])
                n_outputs = len(cmd["outputs"])
                dims = cmd.get("dims", 1)
                func = (
                    f"batch_foreach<kernel_{name}<CpuTypes>, kernel_{name}<SimdTypes>, "
                    f"{n_inputs}, {n_outputs}, {dims}>"
                )
            f.write(
                f"case {opcode}:\n"
                f"    {func}(instr, locals);\n"
                f"    break;\n"
            )


if __name__ == "__main__":
    main()

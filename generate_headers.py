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
            parameters = ", ".join(f"Value {arg['name']}" for arg in cmd["inputs"])
            arguments = ", ".join(arg["name"] for arg in cmd["inputs"])
            instruction_call = f"instruction(\"{name}\", {{{arguments}}})"

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

def instruction_set_mixin(types, commands):
    with open("src/madcode/instruction_set_mixin.h", "w") as f:
        write_autogen(f)
        f.write("using SigType = SimpleInstruction::SigType;\n")

        for name, sig in types.items():
            dtype = "DT_" + sig["dtype"].upper()
            shape = ", ".join(
                str(item) if isinstance(item, int) else f"\"{item}\"" for item in sig["shape"]
            )
            f.write(f"const SimpleInstruction::SigType {name} {{{dtype}, {{{shape}}}}};\n")

        f.write(
            "const auto mi = [](\n"
            "    std::string name,\n"
            "    int opcode,\n"
            "    std::initializer_list<SigType> inputs,\n"
            "    std::initializer_list<SigType> outputs\n"
            ") { return InstructionPtr(new SimpleInstruction(name, opcode, inputs, outputs)); };\n"
            "\n"
            "InstructionPtr instructions[] {\n"
        )
        for name, cmd in commands.items():
            opcode = cmd["opcode"]
            input_types = ", ".join(arg["type"] for arg in cmd["inputs"])
            output_types = ", ".join(ret["type"] for ret in cmd["outputs"])
            f.write(f"    mi(\"{name}\", {opcode}, {{{input_types}}}, {{{output_types}}}),\n")
        f.write("};\n")


def cpu_runtime_mixin(commands):
    with open("src/backend/cpu/runtime_mixin.h", "w") as f:
        write_autogen(f)

        for name, cmd in commands.items():
            opcode = cmd["opcode"]
            n_inputs = len(cmd["inputs"])
            n_outputs = len(cmd["outputs"])
            f.write(
                f"case {opcode}:\n"
                f"    batch_foreach<kernel_{name}, {n_inputs}, {n_outputs}>(instr, locals);\n"
                f"    break;\n"
            )


if __name__ == "__main__":
    main()

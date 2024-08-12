#include <iostream>
#include <fmt/core.h>
#include <fmt/ranges.h>
#include "madevent/madcode.h"
#include "madevent/phasespace.h"

using namespace madevent;

int main() {
    /*const madevent::Type t_five_vector{madevent::DT_FLOAT, {5}};
    //auto instr = madevent::PrintInstruction();
    auto instr = madevent::SimpleInstruction(
        "add",
        {{madevent::DT_FLOAT, {"a"}}, {madevent::DT_FLOAT, {"a"}}},
        {{madevent::DT_FLOAT, {"a"}}}
        //{{madevent::DT_FLOAT, {"..."}}, {madevent::DT_FLOAT, {"..."}}},
        //{{madevent::DT_FLOAT, {"..."}}}
    );
    auto out = instr.signature(std::vector({t_five_vector, t_five_vector}));
    for (auto o : out) {
        fmt::print("{} {}\n", (int)o.dtype, o.shape);
    }*/

    /*FunctionBuilder fb({four_vector, four_vector}, {four_vector});
    auto sum = fb.add(fb.input(0), fb.input(1));
    auto [s, sqrt_s] = fb.s_and_sqrt_s(fb.input(1));
    auto mul = fb.mul_scalar(sum, sqrt_s);
    fb.output(0, mul);
    auto func = fb.function();*/

    auto m1 = TwoParticle(true);
    std::cout << m1.forward_function() << "\n";
    std::cout << m1.inverse_function() << "\n";

    auto m2 = TwoParticle(false);
    std::cout << m2.forward_function() << "\n";
    std::cout << m2.inverse_function() << "\n";

    auto m3 = TInvariantTwoParticle(true);
    std::cout << m3.forward_function() << "\n";
    std::cout << m3.inverse_function() << "\n";

    auto m4 = TInvariantTwoParticle(false);
    std::cout << m4.forward_function() << "\n";
    std::cout << m4.inverse_function() << "\n";

    auto m5 = Luminosity(13000.*13000., 20.*20., 0., 0., 10., 10.);
    std::cout << m5.forward_function() << "\n";
    std::cout << m5.inverse_function() << "\n";

    return 0;
}

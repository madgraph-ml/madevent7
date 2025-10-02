#include "madevent/phasespace/matrix_element.h"

using namespace madevent;

MatrixElement::MatrixElement(
    std::size_t matrix_element_index,
    std::size_t particle_count,
    bool simple_matrix_element,
    std::size_t channel_count
) :
    FunctionGenerator(
        "MatrixElement",
        [&] {
            TypeVec arg_types {
                batch_four_vec_array(particle_count),
                batch_int,
            };
            if (!simple_matrix_element) {
                arg_types.push_back(batch_float);
            }
            return arg_types;
        }(),
        simple_matrix_element ?
            TypeVec{batch_float} :
            TypeVec{batch_float, batch_float_array(channel_count), batch_int, batch_int, batch_int}
    ),
    _matrix_element_index(matrix_element_index),
    _particle_count(particle_count),
    _simple_matrix_element(simple_matrix_element),
    _channel_count(channel_count)
{}

ValueVec MatrixElement::build_function_impl(
    FunctionBuilder& fb, const ValueVec& args
) const {
    auto momenta = args.at(0);
    auto flavor = args.at(1);
    if (_simple_matrix_element) {
        return {fb.matrix_element(momenta, flavor, _matrix_element_index)};
    } else {
        auto alpha_s = args.at(2);
        auto batch_size = fb.batch_size({momenta, alpha_s, flavor});
        auto random = fb.random(batch_size, static_cast<me_int_t>(3));
        auto [me, amp2, diagram_id, color_id, helicity_id] = fb.matrix_element_multichannel(
            momenta, alpha_s, random, flavor, _matrix_element_index, _channel_count
        );
        return {me, amp2, diagram_id, color_id, helicity_id};
    }
}

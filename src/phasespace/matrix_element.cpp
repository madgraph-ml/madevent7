#include "madevent/phasespace/matrix_element.h"

using namespace madevent;

MatrixElement::MatrixElement(
    std::size_t matrix_element_index,
    std::size_t particle_count,
    bool simple_matrix_element,
    std::size_t channel_count,
    const std::vector<int64_t>& amp2_remap
) :
    FunctionGenerator(
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
    _channel_count(channel_count),
    _amp2_remap(amp2_remap)
{}

ValueVec MatrixElement::build_function_impl(
    FunctionBuilder& fb, const ValueVec& args
) const {
    auto momenta = args.at(0);
    auto flavor = args.at(1);
    Value mirror = static_cast<int64_t>(0); //TODO: remove
    if (_simple_matrix_element) {
        return {fb.matrix_element(momenta, flavor, mirror, _matrix_element_index)};
    } else {
        auto alpha_s = args.at(2);
        auto batch_size = fb.batch_size({momenta, alpha_s, flavor, mirror});
        auto random = fb.random(batch_size, static_cast<int64_t>(3));
        auto [me, amp2, diagram_id, color_id, helicity_id] = fb.matrix_element_multichannel(
            momenta, alpha_s, random, flavor, mirror, _matrix_element_index,
            static_cast<int64_t>(_amp2_remap.size())
        );
        auto channel_weights = fb.collect_channel_weights(amp2, _amp2_remap, _channel_count);
        return {me, channel_weights, diagram_id, color_id, helicity_id};
    }
}

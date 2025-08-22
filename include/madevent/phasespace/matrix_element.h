#pragma once

#include "madevent/phasespace/base.h"

namespace madevent {

class MatrixElement : public FunctionGenerator {
public:
    MatrixElement(
        std::size_t matrix_element_index,
        std::size_t particle_count,
        bool simple_matrix_element = true,
        std::size_t channel_count = 1,
        const std::vector<me_int_t>& amp2_remap = {}
    );
    std::size_t channel_count() const { return _channel_count; }
    std::size_t particle_count() const { return _particle_count; }

private:
    ValueVec build_function_impl(FunctionBuilder& fb, const ValueVec& args) const override;

    me_int_t _matrix_element_index;
    std::size_t _particle_count;
    bool _simple_matrix_element;
    me_int_t _channel_count;
    std::vector<me_int_t> _amp2_remap;
};

}

#include "madevent/phasespace/scale.h"

using namespace madevent;

EnergyScale::EnergyScale(
    std::size_t particle_count,
    DynamicScaleType dynamic_scale_type,
    bool ren_scale_fixed,
    bool fact_scale_fixed,
    double ren_scale,
    double fact_scale1,
    double fact_scale2
) :
    FunctionGenerator(
        {batch_four_vec_array(particle_count)},
        {batch_float, batch_float, batch_float}
    ),
    _dynamic_scale_type(dynamic_scale_type),
    _ren_scale_fixed(ren_scale_fixed),
    _fact_scale_fixed(fact_scale_fixed),
    _ren_scale(ren_scale),
    _fact_scale1(fact_scale1),
    _fact_scale2(fact_scale2) {}

ValueVec EnergyScale::build_function_impl(
    FunctionBuilder& fb, const ValueVec& args
) const {
    if (_ren_scale_fixed && _fact_scale_fixed) {
        return {_ren_scale, _fact_scale1, _fact_scale2};
    }
    auto momenta = args.at(0);
    Value scale;
    switch (_dynamic_scale_type) {
    case transverse_energy:
        scale = fb.scale_transverse_energy(momenta);
        break;
    case transverse_mass:
        scale = fb.scale_transverse_mass(momenta);
        break;
    case half_transverse_mass:
        scale = fb.scale_half_transverse_mass(momenta);
        break;
    case partonic_energy:
        scale = fb.scale_partonic_energy(momenta);
        break;
    default:
        throw std::runtime_error("invalid dynamic scale type");
    }
    return {
        _ren_scale_fixed ? _ren_scale : scale,
        _fact_scale_fixed ? _fact_scale1 : scale,
        _fact_scale_fixed ? _fact_scale2 : scale
    };
}

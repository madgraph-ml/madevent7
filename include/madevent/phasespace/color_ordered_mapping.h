#pragma once

#include <vector>

#include "madevent/phasespace/topology.h"
#include "madevent/phasespace/base.h"
#include "madevent/phasespace/invariants.h"
#include "madevent/phasespace/two_particle.h"
#include "madevent/phasespace/three_particle.h"


namespace madevent {

class ColorOrderedMapping : public Mapping {
public:
    ColorOrderedMapping(
        const std::vector<std::size_t>& color_order,
        std::size_t beamA,
        std::size_t beamB,
        double t_invariant_power=0.8,
        double s_invariant_power=0.0
    );
    // Exactly 2 randoms per scheduled step: (t,phi) or (t,s)
    std::size_t random_dim() const {
        return 2 * _steps.size();
    }

private:
    Result build_forward_impl(
        FunctionBuilder& fb, const ValueVec& inputs, const ValueVec& conditions
    ) const override;
    Result build_inverse_impl(
        FunctionBuilder& fb, const ValueVec& inputs, const ValueVec& conditions
    ) const override;

    void init_sets();        // derive _set1, _set2 (ordered) from _color_order + beams
    void build_schedule();   // fill _steps according to sets (central split + peels)

    std::vector<std::size_t> _color_order;
    std::size_t _beamA, _beamB;
    std::vector<std::size_t> _set1, _set2; // ordered lists of particles in each color set

    // Kernels (framing distinguishes first vs subsequent)
    Invariant _uniform_invariant;
    TwoToTwoParticleScattering   _gent_com;   // COM frame 2->2 (first step)
    TwoToTwoParticleScattering   _gent_lab;   // LAB frame 2->2 (first peel in an arc)
    TwoToThreeParticleScattering _gen23;   // LAB frame 2->3 (subsequent peels)

    struct Step {
        enum Type { central_split, peel_arc, peel_line } type;
        std::size_t index; // index in color order of particle being split/peeled
    };
    std::vector<Step> _steps; // schedule of splitting/peeling steps
};

} // namespace madevent

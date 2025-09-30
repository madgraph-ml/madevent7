#pragma once

#include "madevent/phasespace/phasespace.h"
#include "madevent/phasespace/matrix_element.h"
#include "madevent/phasespace/cross_section.h"
#include "madevent/phasespace/vegas.h"
#include "madevent/phasespace/pdf.h"
#include "madevent/phasespace/flow.h"
#include "madevent/phasespace/discrete_sampler.h"
#include "madevent/phasespace/discrete_flow.h"
#include "madevent/phasespace/channel_weights.h"
#include "madevent/phasespace/channel_weight_network.h"
#include "madevent/util.h"

namespace madevent {

class Unweighter : public FunctionGenerator {
public:
    Unweighter(const TypeVec& types);
private:
    ValueVec build_function_impl(FunctionBuilder& fb, const ValueVec& args) const override;
};

class Integrand : public FunctionGenerator {
public:
    using AdaptiveMapping = std::variant<std::monostate, VegasMapping, Flow>;
    using AdaptiveDiscrete = std::variant<std::monostate, DiscreteSampler, DiscreteFlow>;
    inline static const int sample = 1;
    inline static const int unweight = 2;
    inline static const int return_momenta = 4;
    inline static const int return_x1_x2 = 8;
    inline static const int return_random = 16;
    inline static const int return_latent = 32;
    inline static const int return_channel = 64;
    inline static const int return_chan_weights = 128;
    inline static const int return_cwnet_input = 256;
    inline static const int return_discrete = 512;
    inline static const int return_discrete_latent = 1024;

    Integrand(
        const PhaseSpaceMapping& mapping,
        const DifferentialCrossSection& diff_xs,
        const AdaptiveMapping& adaptive_map = std::monostate{},
        const AdaptiveDiscrete& discrete_before = std::monostate{},
        const AdaptiveDiscrete& discrete_after = std::monostate{},
        const std::optional<PdfGrid>& pdf_grid = std::nullopt,
        const std::optional<EnergyScale>& energy_scale = std::nullopt,
        const std::optional<PropagatorChannelWeights>& prop_chan_weights = std::nullopt,
        const std::optional<SubchannelWeights>& subchan_weights = std::nullopt,
        const std::optional<ChannelWeightNetwork>& chan_weight_net = std::nullopt,
        int flags = 0,
        const std::vector<std::size_t>& channel_indices = {},
        const std::vector<std::size_t>& active_flavors = {}
    );
    std::size_t particle_count() const { return _mapping.particle_count(); }
    int flags() const { return _flags; }
    std::optional<std::string> vegas_grid_name() const {
        if (auto vegas = std::get_if<VegasMapping>(&_adaptive_map)) {
            return vegas->grid_name();
        } else {
            return std::nullopt;
        }
    }
    std::size_t vegas_dimension() const {
        if (auto vegas = std::get_if<VegasMapping>(&_adaptive_map)) {
            return vegas->dimension();
        } else {
            return 0;
        }
    }
    std::size_t vegas_bin_count() const {
        if (auto vegas = std::get_if<VegasMapping>(&_adaptive_map)) {
            return vegas->bin_count();
        } else {
            return 0;
        }
    }
    const PhaseSpaceMapping& mapping() const { return _mapping; }
    const DifferentialCrossSection& diff_xs() const { return _diff_xs; }
    const AdaptiveMapping& adaptive_map() const { return _adaptive_map; }
    const AdaptiveDiscrete& discrete_before() const { return _discrete_before; }
    const AdaptiveDiscrete& discrete_after() const { return _discrete_after; }
    const std::optional<EnergyScale>& energy_scale() const { return _energy_scale; }
    const std::optional<PropagatorChannelWeights>& prop_chan_weights() const {
        return _prop_chan_weights;
    }
    const std::optional<ChannelWeightNetwork>& chan_weight_net() const {
        return _chan_weight_net;
    }
    const std::size_t random_dim() const { return _random_dim; }
    std::tuple<std::vector<std::size_t>, std::vector<bool>> latent_dims() const;

private:
    ValueVec build_function_impl(FunctionBuilder& fb, const ValueVec& args) const override;

    PhaseSpaceMapping _mapping;
    DifferentialCrossSection _diff_xs;
    AdaptiveMapping _adaptive_map;
    AdaptiveDiscrete _discrete_before;
    AdaptiveDiscrete _discrete_after;
    std::optional<PartonDensity> _pdf1;
    std::optional<PartonDensity> _pdf2;
    std::vector<me_int_t> _pdf_indices1;
    std::vector<me_int_t> _pdf_indices2;
    std::optional<EnergyScale> _energy_scale;
    std::optional<PropagatorChannelWeights> _prop_chan_weights;
    std::optional<SubchannelWeights> _subchan_weights;
    std::optional<ChannelWeightNetwork> _chan_weight_net;
    int _flags;
    std::vector<me_int_t> _channel_indices;
    me_int_t _random_dim;
    std::size_t _latent_dim;
    std::vector<double> _active_flavors;

    friend class IntegrandProbability;
};

class IntegrandProbability : public FunctionGenerator {
public:
    IntegrandProbability(const Integrand& integrand);

private:
    ValueVec build_function_impl(FunctionBuilder& fb, const ValueVec& args) const override;

    Integrand::AdaptiveMapping _adaptive_map;
    Integrand::AdaptiveDiscrete _discrete_before;
    Integrand::AdaptiveDiscrete _discrete_after;
    std::size_t _permutation_count;
    std::size_t _flavor_count;
    bool _has_pdf_prior;
};

}

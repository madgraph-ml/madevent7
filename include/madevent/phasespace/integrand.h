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
        const std::vector<me_int_t>& chan_weight_remap = {},
        std::size_t remapped_chan_count = 0,
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
    struct ChannelArgs {
        Value r, batch_size;
        bool has_permutations, has_multi_flavor, has_mirror, has_pdf_prior;
        Value max_weight;
    };
    struct ChannelResult {
        Value r, latent;
        Value momenta, momenta_mirror, momenta_acc;
        Value x1, x1_acc;
        Value x2, x2_acc;
        Value pdf_prior;
        Value chan_index, chan_index_in_group, flavor_id, mirror_id, indices_acc;
        Value weight_before_cuts, weight_after_cuts, adaptive_prob;
        ValueVec xs_cache;
    };

    ValueVec build_function_impl(FunctionBuilder& fb, const ValueVec& args) const override;
    ChannelResult build_channel_part(FunctionBuilder& fb, const ChannelArgs& args) const;
    ValueVec build_common_part(
        FunctionBuilder& fb, const ChannelArgs& args, const ChannelResult& result
    ) const;

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
    std::vector<me_int_t> _chan_weight_remap;
    me_int_t _remapped_chan_count;
    int _flags;
    std::vector<me_int_t> _channel_indices;
    me_int_t _random_dim;
    std::size_t _latent_dim;
    std::vector<double> _active_flavors;

    friend class IntegrandProbability;
    friend class MultiChannelIntegrand;
};

class MultiChannelIntegrand : public FunctionGenerator {
public:
    MultiChannelIntegrand(const std::vector<std::shared_ptr<Integrand>>& integrands);

private:
    ValueVec build_function_impl(FunctionBuilder& fb, const ValueVec& args) const override;

    std::vector<std::shared_ptr<Integrand>> _integrands;
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

#include "madevent/phasespace/cross_section.h"

#include <set>

using namespace madevent;

DifferentialCrossSection::DifferentialCrossSection(
    const std::vector<std::vector<int64_t>>& pid_options,
    std::size_t matrix_element_index,
    const RunningCoupling& running_coupling,
    const std::optional<PdfGrid>& pdf_grid,
    double cm_energy,
    const EnergyScale& energy_scale,
    bool simple_matrix_element,
    std::size_t channel_count,
    const std::vector<int64_t>& amp2_remap,
    bool has_mirror
) :
    FunctionGenerator(
        "DifferentialCrossSection",
        [&] {
            TypeVec arg_types {
                batch_four_vec_array(pid_options.at(0).size()),
                batch_float,
                batch_float,
                batch_int
            };
            if (has_mirror) {
                arg_types.push_back(batch_int);
            }
            if (!pdf_grid) {
                std::set<int> pids1, pids2;
                for (auto& option : pid_options) {
                    pids1.insert(option.at(0));
                    pids2.insert(option.at(1));
                }
                arg_types.push_back(batch_float_array(pids1.size()));
                arg_types.push_back(batch_float_array(pids2.size()));
                arg_types.push_back(batch_float);
            }
            return arg_types;
        }(),
        simple_matrix_element ?
            TypeVec{batch_float} :
            TypeVec{
                batch_float, batch_float_array(channel_count),
                batch_int, batch_int, batch_int
            }
    ),
    _pid_options(pid_options),
    _matrix_element(
        matrix_element_index,
        pid_options.at(0).size(),
        simple_matrix_element,
        channel_count,
        amp2_remap
    ),
    _running_coupling(running_coupling),
    _e_cm2(cm_energy * cm_energy),
    _energy_scale(energy_scale),
    _simple_matrix_element(simple_matrix_element),
    _has_mirror(has_mirror),
    _channel_count(channel_count),
    _amp2_remap(amp2_remap)
{
    if (pdf_grid) {
        std::vector<int> pids1, pids2;
        for (auto& option : pid_options) {
            pids1.push_back(option.at(0));
            pids2.push_back(option.at(1));
        }
        _pdf1 = PartonDensity(pdf_grid.value(), pids1, true);
        _pdf2 = PartonDensity(pdf_grid.value(), pids2, true);
    } else {
        std::set<int> pids1, pids2;
        for (auto& option : pid_options) {
            pids1.insert(option.at(0));
            pids2.insert(option.at(1));
        }
        for (auto& option : pid_options) {
            _pdf_indices1.push_back(std::distance(pids1.begin(), pids1.find(option.at(0))));
            _pdf_indices2.push_back(std::distance(pids2.begin(), pids2.find(option.at(1))));
        }
    }
}

ValueVec DifferentialCrossSection::build_function_impl(
    FunctionBuilder& fb, const ValueVec& args
) const {
    auto momenta = args.at(0);
    auto x1 = args.at(1);
    auto x2 = args.at(2);
    auto flavor_id = args.at(3);
    //auto mirror_id = args.at(4);
    std::size_t arg_index = 4;
    if (_has_mirror) ++arg_index;
    //TODO: need to use mirror_id if we have two different PDFs

    Value pdf1, pdf2, ren_scale;
    if (_pdf1) {
        auto scales = _energy_scale.build_function(fb, {momenta});
        pdf1 = _pdf1.value().build_function(
            fb, {x1, scales.at(1), flavor_id}
        ).at(0);
        pdf2 = _pdf2.value().build_function(
            fb, {x2, scales.at(2), flavor_id}
        ).at(0);
        ren_scale = scales.at(0);
    } else {
        pdf1 = fb.gather(fb.gather_int(flavor_id, _pdf_indices1), args.at(arg_index));
        pdf2 = fb.gather(fb.gather_int(flavor_id, _pdf_indices2), args.at(arg_index + 1));
        ren_scale = args.at(arg_index + 2);
    }

    if (_simple_matrix_element) {
        auto me_result = _matrix_element.build_function(fb, {momenta, flavor_id});
        return {fb.diff_cross_section(x1, x2, pdf1, pdf2, me_result.at(0), _e_cm2)};
    } else {
        auto alpha_s = _running_coupling.build_function(fb, {ren_scale}).at(0);
        Value me2, chan_weights, color_id, diagram_id;
        auto me_result = _matrix_element.build_function(
            fb, {momenta, flavor_id, alpha_s}
        );
        me_result.at(0) = fb.diff_cross_section(x1, x2, pdf1, pdf2, me_result.at(0), _e_cm2);
        return me_result;
    }
}


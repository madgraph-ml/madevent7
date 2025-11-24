#pragma once

#include "madevent/phasespace/matrix_element.h"
#include "madevent/phasespace/pdf.h"
#include "madevent/phasespace/scale.h"

namespace madevent {

class DifferentialCrossSection : public FunctionGenerator {
public:
    DifferentialCrossSection(
        const std::vector<std::vector<me_int_t>>& pid_options,
        std::size_t matrix_element_index,
        const RunningCoupling& running_coupling,
        const std::optional<PdfGrid>& pdf_grid,
        double cm_energy,
        const EnergyScale& energy_scale,
        bool simple_matrix_element = true,
        std::size_t channel_count = 1,
        bool has_mirror = false
    );
    const std::vector<std::vector<me_int_t>>& pid_options() const {
        return _pid_options;
    }
    std::size_t channel_count() const { return _channel_count; }
    bool has_mirror() const { return _has_mirror; }
    const MatrixElement& matrix_element() const { return _matrix_element; }

private:
    ValueVec
    build_function_impl(FunctionBuilder& fb, const ValueVec& args) const override;

    std::vector<std::vector<me_int_t>> _pid_options;
    MatrixElement _matrix_element;
    std::optional<PartonDensity> _pdf1;
    std::optional<PartonDensity> _pdf2;
    std::vector<me_int_t> _pdf_indices1;
    std::vector<me_int_t> _pdf_indices2;
    RunningCoupling _running_coupling;
    double _e_cm2;
    EnergyScale _energy_scale;
    bool _simple_matrix_element;
    bool _has_mirror;
    me_int_t _channel_count;
    std::vector<me_int_t> _amp2_remap;
};

} // namespace madevent

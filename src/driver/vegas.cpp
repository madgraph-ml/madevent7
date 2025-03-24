#include "madevent/driver/vegas.h"

using namespace madevent;

void VegasGridOptimizer::optimize(Tensor weights, Tensor inputs) {
    auto grid = _context->global(_grid_name);
    auto grid_cpu = grid.cpu();
    auto weights_cpu = weights.cpu();
    auto inputs_cpu = inputs.cpu();

    std::size_t n_samples = inputs.size(0);
    std::size_t n_dims = inputs.size(1);
    std::size_t n_bins = grid_cpu.size(2) - 1;
    //TODO: check all the shapes here

    auto new_grid = grid_cpu.copy();
    std::vector<std::size_t> bin_counts(n_bins);
    std::vector<double> bin_values(n_bins);

    auto grid_view = grid_cpu.view<double, 3>()[0];
    auto new_grid_view = new_grid.view<double, 3>()[0];
    auto w_view = weights_cpu.view<double, 1>();
    auto in_view = inputs_cpu.view<double, 2>();

    for (std::size_t i_dim = 0; i_dim < n_dims; ++i_dim) {
        std::fill(bin_counts.begin(), bin_counts.end(), 0);
        std::fill(bin_values.begin(), bin_values.end(), 0.);

        // build histograms
        for (std::size_t i_sample = 0; i_sample < n_samples; ++i_sample) {
            int i_bin = in_view[i_sample][i_dim] * n_bins;
            if (i_bin < 0 || i_bin >= n_bins) continue;
            double w = w_view[i_sample];
            bin_values[i_bin] += w * w;
            ++bin_counts[i_bin];
        }

        // compute averages
        for (std::size_t i_bin = 0; i_bin < n_bins; ++i_bin) {
            if (bin_counts[i_bin] > 0) bin_values[i_bin] /= bin_counts[i_bin];
        }

        // apply smoothing
        double prev_value = bin_values[0];
        double current_value = bin_values[1];
        double sum = 0.;
        bin_values[0] = (7. * prev_value + current_value) / 8.;
        for (std::size_t i_bin = 1; i_bin < n_bins - 1; ++i_bin) {
            double next_value = bin_values[i_bin + 1];
            double new_value = (prev_value + 6. * current_value + next_value) / 8.;
            bin_values[i_bin] = new_value;
            sum += new_value;
            prev_value = current_value;
            current_value = next_value;
        }
        bin_values[n_bins - 1] = (prev_value + 7. * current_value) / 8.;

        // normalize and apply damping
        constexpr double tiny = 1e-257;
        double damped_avg = 0.;
        if (sum == 0) {
            std::fill(
                bin_values.begin(),
                bin_values.end(),
                std::pow(-(1 - tiny) / std::log(tiny), _damping)
            );
            damped_avg = tiny;
        } else {
            for (std::size_t i_bin = 0; i_bin < n_bins; ++i_bin) {
                double val_norm = bin_values[i_bin] / sum + tiny;
                double new_val = val_norm <= 0.99999999 ?
                    std::pow(-(1 - val_norm) / std::log(val_norm), _damping) :
                    val_norm;
                bin_values[i_bin] = new_val;
                damped_avg += new_val;
            }
            damped_avg /= n_bins;
        }

        // update grid
        double accumulator = 0.;
        int64_t j_bin = -1;
        for (std::size_t i_bin = 1; i_bin < n_bins; ++i_bin) {
            while (accumulator < damped_avg) {
                ++j_bin;
                if (j_bin == n_bins) break;
                accumulator += bin_values[j_bin];
            }
            if (j_bin == n_bins) break;
            double grid_j = grid_view[i_dim][j_bin];
            double grid_j_next = grid_view[i_dim][j_bin + 1];
            double bin_width = grid_j_next - grid_j;
            accumulator -= damped_avg;
            new_grid_view[i_dim][i_bin] = grid_j_next - accumulator / bin_values[j_bin] * bin_width;
        }
    }

    grid.copy_from(new_grid);
}

std::size_t VegasGridOptimizer::input_dim() const {
    return _context->global(_grid_name).size(1);
}

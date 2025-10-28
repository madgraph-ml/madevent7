#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include <fstream>
#include <random>

#include "madevent/phasespace/topology.h"

namespace madevent {

struct LHEHeader {
    std::string name;
    std::string content;
    bool escape_content;
};

struct LHEProcess {
    double cross_section;
    double cross_section_error;
    double max_weight;
    int process_id;
};

struct LHEMeta {
    int beam1_pdg_id, beam2_pdg_id;
    double beam1_energy, beam2_energy;
    int beam1_pdf_authors, beam2_pdf_authors;
    int beam1_pdf_id, beam2_pdf_id;
    int weight_mode;
    std::vector<LHEProcess> processes;
    std::vector<LHEHeader> headers;
};

struct LHEParticle {
    // particle-level information as defined in arXiv:0109068
    inline static const int status_incoming = -1;
    inline static const int status_outgoing = 1;
    inline static const int status_intermediate_resonance = 2;

    int pdg_id;
    int status_code;
    int mother1, mother2;
    int color, anti_color;
    double p_x, p_y, p_z, energy, mass;
    double lifetime;
    double spin;
};

struct LHEEvent {
    // event-level information as defined in arXiv:0109068
    int process_id;
    double weight;
    double scale;
    double alpha_qed;
    double alpha_qcd;
    std::vector<LHEParticle> particles;
};

class LHECompleter {
public:
    struct SubprocArgs {
        int process_id;
        std::vector<Topology> topologies;
        std::vector<std::vector<std::vector<std::size_t>>> permutations;
        std::vector<std::vector<std::size_t>> channel_indices;
        std::vector<std::vector<std::tuple<int, int>>> color_flows;
        std::unordered_map<int, int> pdg_color_types;
        std::vector<std::vector<double>> helicities;
        std::vector<std::vector<std::vector<int>>> pdg_ids;
    };

    LHECompleter(const std::vector<SubprocArgs>& subproc_args, double bw_cutoff);
    void complete_event_data(
        LHEEvent& event,
        int subprocess_index,
        int diagram_index,
        int color_index,
        int flavor_index,
        int helicity_index
    );
    std::size_t max_particle_count() const { return _max_particle_count; }

private:
    struct SubprocData {
        int process_id;
        std::size_t color_offset, pdg_id_offset, helicity_offset, mass_offset;
        std::size_t particle_count, color_count, flavor_count, diagram_count, helicity_count;
    };
    struct PropagatorData {
        int color, anti_color;
        int pdg_id;
        int momentum_mask;
        double mass, width;
    };
    struct ResonanceData {
        std::vector<PropagatorData> propagators;
        std::vector<std::tuple<int, int>> mothers;
    };
    std::vector<SubprocData> _subproc_data; // n_subproc
    std::vector<int> _process_indices; // n_subproc
    std::vector<double> _masses; // sum_i n_particles_i
    std::vector<std::tuple<int, int>> _colors; // sum_i n_colors_i * n_particles_i
    std::vector<double> _helicities; // sum_i n_hel_i * n_particles_i
    std::vector<std::tuple<std::size_t, std::size_t>> _pdg_id_index_and_count; // n_pids
    std::vector<int> _pdg_ids; // sum n_pids_all * n_particles
    std::unordered_map<std::size_t, std::vector<ResonanceData>> _resonance_data; // n_diagrams
    double _bw_cutoff;
    std::size_t _max_particle_count;
    std::mt19937 _rand_gen;
};

class LHEFileWriter {
public:
    LHEFileWriter(
        const std::string& file_name, const LHEMeta& meta
    );
    void write(const LHEEvent& event);
    ~LHEFileWriter();

private:
    std::ofstream _file_stream;
};

}

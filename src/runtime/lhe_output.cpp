#include "madevent/runtime/lhe_output.h"

#include "madevent/util.h"

#include <span>

using namespace madevent;

namespace {

std::size_t cantor_pairing(std::size_t i, std::size_t j) {
    return (i + j) * (i + j + 1) / 2 + i;
}

std::size_t cantor_pairing(std::size_t i, std::size_t j, std::size_t k) {
    return cantor_pairing(cantor_pairing(i, j), k);
}

}

LHECompleter::LHECompleter(
    const std::vector<SubprocArgs>& subproc_args, double bw_cutoff
) :
    _bw_cutoff(bw_cutoff),
    _max_particle_count(0),
    _rand_gen(std::random_device{}())
{
    std::size_t color_offset = 0, pdg_id_offset = 0, helicity_offset = 0;
    _max_particle_count = 0;
    for (auto& args : subproc_args) {
        std::size_t particle_count = args.helicities.at(0).size();
        if (_max_particle_count < particle_count) _max_particle_count = particle_count;

        for (auto& helicities : args.helicities) {
            if (helicities.size() != particle_count) {
                throw std::invalid_argument("Invalid number of helicities");
            }
            _helicities.insert(_helicities.end(), helicities.begin(), helicities.end());
        }

        for (auto& color_flows : args.color_flows) {
            if (color_flows.size() != particle_count) {
                throw std::invalid_argument("Invalid number of colors");
            }
            _colors.insert(_colors.end(), color_flows.begin(), color_flows.end());
        }

        for (auto& pdg_id_options : args.pdg_ids) {
            if (pdg_id_options.size() == 0) {
                throw std::invalid_argument(
                    "Must provide at least one option per flavor index"
                );
            }
            _pdg_id_index_and_count.push_back({_pdg_ids.size(), pdg_id_options.size()});
            for (auto& pdg_ids : pdg_id_options) {
                if (pdg_ids.size() != particle_count) {
                    throw std::invalid_argument("Invalid number of particles ids");
                }
                _pdg_ids.insert(_pdg_ids.end(), pdg_ids.begin(), pdg_ids.end());
            }
        }

        std::size_t diagram_count = 0;
        for (auto& color_flows : args.color_flows) {
            if (color_flows.size() != particle_count) {
                throw std::invalid_argument("Invalid number of colors");
            }
            _colors.insert(_colors.end(), color_flows.begin(), color_flows.end());
        }

        _subproc_data.push_back({
            .process_id = args.process_id,
            .color_offset = color_offset,
            .pdg_id_offset = pdg_id_offset,
            .helicity_offset = helicity_offset,
            .particle_count = particle_count,
            .color_count = args.color_flows.size(),
            .flavor_count = args.pdg_ids.size(),
            .diagram_count = diagram_count,
            .helicity_count = args.helicities.size(),
        });

        helicity_offset += particle_count * args.helicities.size();
        color_offset += particle_count * args.color_flows.size();
        pdg_id_offset += args.pdg_ids.size();
    }
}

void LHECompleter::complete_event_data(
    LHEEvent& event,
    int subprocess_index,
    int diagram_index,
    int color_index,
    int flavor_index,
    int helicity_index
) {
    auto& subproc_data = _subproc_data.at(subprocess_index);
    if (event.particles.size() != subproc_data.particle_count) {
        throw std::runtime_error("Invalid particle number for subprocess");
    }
    //if (diagram_index < 0 || diagram_index >= subproc_data.diagram_count) {
    //    throw std::runtime_error("Diagram index out of range");
    //}
    if (color_index < 0 || color_index >= subproc_data.color_count) {
        throw std::runtime_error("Color index out of range");
    }
    if (flavor_index < 0 || flavor_index >= subproc_data.flavor_count) {
        throw std::runtime_error("Flavor index out of range");
    }
    if (helicity_index < 0 || helicity_index >= subproc_data.helicity_count) {
        throw std::runtime_error("Helicity index out of range");
    }

    event.process_id = subproc_data.process_id;

    std::size_t color_offset =
        subproc_data.color_offset + subproc_data.particle_count * color_index;
    std::size_t helicity_offset =
        subproc_data.helicity_offset + subproc_data.particle_count * helicity_index;

    auto [pdg_index, pdg_count] = _pdg_id_index_and_count.at(
        subproc_data.pdg_id_offset + flavor_index
    );
    std::uniform_int_distribution<std::size_t> dist(0, pdg_count - 1);
    std::size_t pdg_random = dist(_rand_gen);
    std::size_t pdg_offset = pdg_index + subproc_data.particle_count * pdg_random;

    for (std::size_t particle_index = 0; auto& particle : event.particles) {
        std::tie(particle.color, particle.anti_color) =
            _colors.at(color_offset + particle_index);
        particle.pdg_id = _pdg_ids.at(pdg_offset + particle_index);
        if (particle_index < 2) {
            particle.status_code = -1;
            particle.mother1 = 0;
            particle.mother2 = 0;
        } else {
            particle.status_code = 1;
            particle.mother1 = 1;
            particle.mother2 = 2;
        }
        particle.lifetime = 0;
        particle.spin = _helicities.at(helicity_offset + particle_index);
        ++particle_index;
    }

    auto find_resonance = _resonance_data.find(cantor_pairing(
        subprocess_index, diagram_index, color_index
    ));
    if (find_resonance == _resonance_data.end()) return;
    auto& resonance_data = find_resonance->second;
    std::vector<std::array<double, 4>> prop_momenta;
    for (auto& resonance_option : resonance_data) {
        bool found_resonances = true;
        prop_momenta.clear();
        for (auto& propagator : resonance_option.propagators) {
            int momentum_mask = propagator.momentum_mask;
            double e, px, py, pz;
            for (auto& particle : event.particles) {
                if (momentum_mask & 1) {
                    e += particle.energy;
                    px += particle.p_x;
                    py += particle.p_y;
                    pz += particle.p_z;
                }
                momentum_mask <<= 1;
            }
            double m2 = e * e - px * px - py * py - pz * pz;
            double m_min = propagator.mass - _bw_cutoff * propagator.width;
            double m_max = propagator.mass + _bw_cutoff * propagator.width;
            if (m2 < m_min * m_min || m2 > m_max * m_max) {
                found_resonances = false;
                break;
            }
            prop_momenta.push_back({e, px, py, pz});
        }
        if (!found_resonances) continue;
        std::size_t prop_count = resonance_option.propagators.size();
        event.particles.insert(event.particles.begin() + 2, prop_count, {});
        for (auto [particle, propagator, momentum] : zip(
            std::span(event.particles.begin() + 2, event.particles.begin() + 2 + prop_count),
            resonance_option.propagators,
            prop_momenta
        )) {
            particle.pdg_id = propagator.pdg_id;
            particle.status_code = 2;
            particle.color = propagator.color;
            particle.anti_color = propagator.anti_color;
            particle.energy = momentum[0];
            particle.p_x = momentum[1];
            particle.p_y = momentum[2];
            particle.p_z = momentum[3];
            particle.mass = std::sqrt(
                particle.energy * particle.energy +
                particle.p_x * particle.p_x +
                particle.p_y * particle.p_y +
                particle.p_z * particle.p_z
            );
            particle.lifetime = 0;
            particle.spin = 9;
        }
        for (auto [particle, mothers] : zip(event.particles, resonance_option.mothers)) {
            std::tie(particle.mother1, particle.mother2) = mothers;
        }
        break;
    }
}

LHEFileWriter::LHEFileWriter(
    const std::string& file_name,
    const LHEMeta& meta
) :
    _file_stream(file_name)
{
    _file_stream << "<LesHouchesEvents version=\"3.0\">\n<header>\n";
    for (auto [name, content, escape_content] : meta.headers) {
        _file_stream << (
            escape_content ?
            std::format("<{0}>\n<![CDATA[\n{1}\n]]>\n</{0}>\n", name, content) :
            std::format("<{0}>\n{1}\n</{0}>\n", name, content)
        );
    }
    _file_stream << std::format(
        "</header>\n<init>\n{} {} {} {} {} {} {} {} {} {}\n",
        meta.beam1_pdg_id, meta.beam2_pdg_id,
        meta.beam1_energy, meta.beam2_energy,
        meta.beam1_pdf_authors, meta.beam2_pdf_authors,
        meta.beam1_pdf_id, meta.beam2_pdf_id,
        meta.weight_mode, meta.processes.size()
    );
    for (auto process : meta.processes) {
        _file_stream << std::format(
            "{} {} {} {}\n",
            process.cross_section,
            process.cross_section_error,
            process.max_weight,
            process.process_id
        );
    }
    _file_stream << "</init>\n";
}

void LHEFileWriter::write(const LHEEvent& event) {
    _file_stream << std::format(
        "<event>\n{} {} {} {} {} {}\n",
        event.particles.size(),
        event.process_id,
        event.weight,
        event.scale,
        event.alpha_qed,
        event.alpha_qcd
    );
    for (auto particle : event.particles) {
        _file_stream << std::format(
            "{} {} {} {} {} {} {} {} {} {} {} {} {}\n",
            particle.pdg_id, particle.status_code,
            particle.mother1, particle.mother2,
            particle.color, particle.anti_color,
            particle.p_x, particle.p_y, particle.p_z, particle.energy, particle.mass,
            particle.lifetime,
            particle.spin
        );
    }
    _file_stream << "</event>\n";
}

LHEFileWriter::~LHEFileWriter() {
    _file_stream << "</LesHouchesEvents>\n";
}

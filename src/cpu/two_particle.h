#pragma once

#include <cmath>

namespace madevent {
namespace cpu {

const double EPS = 1e-12;
const double EPS2 = 1e-24;


// Helper functions

inline double _kaellen(double x, double y, double z) {
    return (x - y - z) ** 2 - 4 * y * z;
}

// Kernels

def decay_momentum(s: Tensor, sqrt_s: Tensor, m1: Tensor, m2: Tensor) -> Tensor2:
    p = torch.zeros((s.shape[0], 4), dtype=s.dtype, device=s.device)
    sqrt_kaellen = kin.kaellen(s, m1**2, m2**2).sqrt()
    p[:, 0] = (s + m1**2 - m2**2) / (2 * sqrt_s)
    p[:, 3] = sqrt_kaellen / (2 * sqrt_s)
    gs = sqrt_kaellen / (2 * s) * pi
    return p, gs

def invt_min_max(s: Tensor, s_in1: Tensor, s_in2: Tensor, m1: Tensor, m2: Tensor) -> Tensor2:
    cos_min = (-1.0) * torch.ones_like(s)
    cos_max = (+1.0) * torch.ones_like(s)
    t_min = -tc.costheta_to_invt(s, s_in1, s_in2, m1, m2, cos_max)
    t_max = -tc.costheta_to_invt(s, s_in1, s_in2, m1, m2, cos_min)
    return t_min, t_max

def invt_to_costheta(
    s: Tensor, s_in1: Tensor, s_in2: Tensor, m1: Tensor, m2: Tensor, t: Tensor
) -> Tensor:
    return tc.invt_to_costheta(s, s_in1, s_in2, m1, m2, -t)

def tinv_two_particle_density(det_t: Tensor, s: Tensor, s_in1: Tensor, s_in2: Tensor) -> Tensor:
    return det_t * pi / (2 * kin.kaellen(s, s_in1, s_in2).sqrt())

}
}

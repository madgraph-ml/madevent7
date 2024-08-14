// Helper functions

double _kaellen(double x, double y, double z) {
    auto xyz = x - y - z;
    return xyz * xyz - 4 * y * z;
}

// Kernels

KERNELSPEC void kernel_decay_momentum(DoubleInput s, DoubleInput sqrt_s, DoubleInput m1, DoubleInput m2, DoubleOutput p) {
    /*p = torch.zeros((s.shape[0], 4), dtype=s.dtype, device=s.device)
    sqrt_kaellen = kin.kaellen(s, m1**2, m2**2).sqrt()
    p[:, 0] = (s + m1**2 - m2**2) / (2 * sqrt_s)
    p[:, 3] = sqrt_kaellen / (2 * sqrt_s)
    gs = sqrt_kaellen / (2 * s) * pi
    return p, gs*/
}

KERNELSPEC void kernel_invt_min_max(
    DoubleInput s, DoubleInput s_in1, DoubleInput s_in2, DoubleInput m1, DoubleInput m2, DoubleOutput t_min, DoubleOutput t_max
) {
    /*cos_min = (-1.0) * torch.ones_like(s)
    cos_max = (+1.0) * torch.ones_like(s)
    t_min = -tc.costheta_to_invt(s, s_in1, s_in2, m1, m2, cos_max)
    t_max = -tc.costheta_to_invt(s, s_in1, s_in2, m1, m2, cos_min)
    return t_min, t_max*/
}

KERNELSPEC void kernel_invt_to_costheta(
    DoubleInput s, DoubleInput s_in1, DoubleInput s_in2, DoubleInput m1, DoubleInput m2, DoubleInput t, DoubleOutput cos_theta
) {
    //return tc.invt_to_costheta(s, s_in1, s_in2, m1, m2, -t)
}

KERNELSPEC void kernel_costheta_to_invt(
    DoubleInput s, DoubleInput s_in1, DoubleInput s_in2, DoubleInput m1, DoubleInput m2, DoubleInput cos_theta, DoubleOutput t
) {
    //return tc.invt_to_costheta(s, s_in1, s_in2, m1, m2, -t)
}

KERNELSPEC void kernel_two_particle_density(DoubleInput s, DoubleInput m1, DoubleInput m2, DoubleOutput gs) {

}

KERNELSPEC void kernel_two_particle_density_inverse(DoubleInput s, DoubleInput m1, DoubleInput m2, DoubleOutput gs) {

}

KERNELSPEC void kernel_tinv_two_particle_density(
    DoubleInput det_t, DoubleInput s, DoubleInput s_in1, DoubleInput s_in2, DoubleOutput det
) {
    //return det_t * pi / (2 * kin.kaellen(s, s_in1, s_in2).sqrt())
}

KERNELSPEC void kernel_tinv_two_particle_density_inverse(
    DoubleInput det_t, DoubleInput s, DoubleInput s_in1, DoubleInput s_in2, DoubleOutput det
) {

}

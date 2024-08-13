// Helper functions

inline double _kaellen(double x, double y, double z) {
    return (x - y - z) ** 2 - 4 * y * z;
}

// Kernels

KERNELSPEC void decay_momentum(double s, double sqrt_s, double m1, double m2, Accessor p) {
    /*p = torch.zeros((s.shape[0], 4), dtype=s.dtype, device=s.device)
    sqrt_kaellen = kin.kaellen(s, m1**2, m2**2).sqrt()
    p[:, 0] = (s + m1**2 - m2**2) / (2 * sqrt_s)
    p[:, 3] = sqrt_kaellen / (2 * sqrt_s)
    gs = sqrt_kaellen / (2 * s) * pi
    return p, gs*/
}

KERNELSPEC void invt_min_max(
    double s, double s_in1, double s_in2, double m1, double m2, double& t_min, double& t_max
) {
    /*cos_min = (-1.0) * torch.ones_like(s)
    cos_max = (+1.0) * torch.ones_like(s)
    t_min = -tc.costheta_to_invt(s, s_in1, s_in2, m1, m2, cos_max)
    t_max = -tc.costheta_to_invt(s, s_in1, s_in2, m1, m2, cos_min)
    return t_min, t_max*/
}

KERNELSPEC void invt_to_costheta(
    double s, double s_in1, double s_in2, double m1, double m2, double t, double& cos_theta
) {
    //return tc.invt_to_costheta(s, s_in1, s_in2, m1, m2, -t)
}

KERNELSPEC void costheta_to_invt(
    double s, double s_in1, double s_in2, double m1, double m2, double cos_theta, double& t
) {
    //return tc.invt_to_costheta(s, s_in1, s_in2, m1, m2, -t)
}

KERNELSPEC void two_particle_density(double s, double m1, double m2, double& gs) {

}

KERNELSPEC void inverse_two_particle_density(double s, double m1, double m2, double& gs) {

}

KERNELSPEC void tinv_two_particle_density(
    double det_t, double s, double s_in1, double s_in2, double& det
) {
    //return det_t * pi / (2 * kin.kaellen(s, s_in1, s_in2).sqrt())
}

KERNELSPEC void tinv_two_particle_density_inverse(
    double det_t, double s, double s_in1, double s_in2, double& det
) {

}

import pytest
from pytest import approx
import madevent7 as me
import numpy as np
from collections import namedtuple

@pytest.fixture
def rng():
    return np.random.default_rng(1234)

@pytest.fixture(params=[
    {"decay": True, "com": False},
    {"decay": True, "com": True},
    {"decay": False, "com": False},
    {"decay": False, "com": True},
], ids=[
    "1->2 decay",
    "1->2 decay, COM",
    "2->2 scattering",
    "2->2 scattering, COM",
])
def mapping_and_args(request):
    com = request.param["com"]
    zeros = np.zeros(N)
    if request.param["decay"]:
        mapping = me.TwoBodyDecay(com=com)
        if com:
            def make_args(point):
                p0 = np.stack([point.m0, zeros, zeros, zeros], axis=1)
                return (
                    [point.r1, point.r2, point.m0, point.m1, point.m2],
                    [],
                    p0
                )
        else:
            def make_args(point):
                return (
                    [point.r1, point.r2, point.m0, point.m1, point.m2, point.p0],
                    [],
                    point.p0
                )
    else:
        mapping = me.TwoToTwoParticleScattering(com=com)
        if com:
            def make_args(point):
                e0 = point.m0 / 2
                p0 = np.stack([point.m0, zeros, zeros, zeros], axis=1)
                pa = np.stack([e0, zeros, zeros, e0], axis=1)
                pb = np.stack([e0, zeros, zeros, -e0], axis=1)
                return [point.r1, point.r2, point.m1, point.m2], [pa, pb], p0
        else:
            def make_args(point):
                return (
                    [point.r1, point.r2, point.m1, point.m2],
                    [point.pa, point.pb],
                    point.p0
                )
    return (mapping, make_args)

def mass(momentum):
    return np.sqrt(momentum[:, 0]**2 - np.sum(momentum[:, 1:]**2, axis=1))

InputPoint = namedtuple("InputPoint", ["r1", "r2", "m0", "m1", "m2", "p0", "pa", "pb"])

N = 10000

ZEROS = np.zeros(N)
PA = np.full((N, 4), [np.sqrt(100**2 + 200**2 + 300**2), 100., 200., 300.])
PB = np.full((N, 4), [np.sqrt(100**2 + 200**2 + 400**2), 100., 200., -400.])
P0 = PA + PB
M0 = mass(P0)
M1 = np.full(N, 20.)
M2 = np.full(N, 30.)

@pytest.fixture(params=[
    (M0, ZEROS, ZEROS, P0, PA, PB),
    (M0, M1, ZEROS, P0, PA, PB),
    (M0, M1, M2, P0, PA, PB),
], ids=[
    "both massless",
    "one massive",
    "both massive",
])
def fixed_input_points(request, rng):
    r1 = rng.random(N)
    r2 = rng.random(N)
    return InputPoint(r1, r2, *request.param)

@pytest.fixture
def input_points(rng):
    max_mass = 125.
    r1 = rng.random(N)
    r2 = rng.random(N)

    pa = np.array([[0, 0, 4000.]]) + rng.normal(0., 500., (N, 3))
    ma = rng.uniform(2.5 * max_mass, 1000., N)
    ea = np.sqrt(np.sum(pa**2, axis=1) + ma ** 2)
    pa = np.concatenate([ea[:,None], pa], axis=1)

    pb = np.array([[0, 0, -4000.]]) + rng.normal(0., 500., (N, 3))
    mb = rng.uniform(20., 500., N)
    eb = np.sqrt(np.sum(pb**2, axis=1) + mb ** 2)
    pb = np.concatenate([eb[:,None], pb], axis=1)

    p0 = pa + pb
    m0 = mass(p0)

    m1 = rng.uniform(1., max_mass, N)
    m2 = rng.uniform(1., max_mass, N)

    return InputPoint(r1, r2, m0, m1, m2, p0, pa, pb)


def test_momentum_conservation(mapping_and_args, input_points):
    mapping, make_args = mapping_and_args
    inputs, conditions, p0 = make_args(input_points)
    (p1, p2), det = mapping.map_forward(inputs, conditions)
    assert p1 + p2 == approx(p0)


def test_outgoing_masses(mapping_and_args, input_points):
    mapping, make_args = mapping_and_args
    inputs, conditions, p0 = make_args(input_points)
    (p1, p2), det = mapping.map_forward(inputs, conditions)
    m0 = mass(p1 + p2)
    m1 = mass(p1)
    m2 = mass(p2)
    assert m0 == approx(input_points.m0)
    assert m1 == approx(input_points.m1)
    assert m2 == approx(input_points.m2)

def test_phase_space_volume(mapping_and_args, fixed_input_points):
    mapping, make_args = mapping_and_args
    inputs, conditions, p0 = make_args(fixed_input_points)
    (p1, p2), det = mapping.map_forward(inputs, conditions)

    s = fixed_input_points.m0 ** 2
    m1_2 = fixed_input_points.m1 ** 2
    m2_2 = fixed_input_points.m2 ** 2
    phase_space_vol = np.pi / (2 * s) * np.sqrt((s - m1_2 - m2_2)**2 - 4 * m1_2 * m2_2)

    assert np.mean(det) == approx(phase_space_vol)


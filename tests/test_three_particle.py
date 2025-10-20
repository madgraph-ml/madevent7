import pytest
from pytest import approx
import madevent7 as me
import numpy as np
from collections import namedtuple


@pytest.fixture
def rng():
    return np.random.default_rng(1234)


@pytest.fixture(
    params=[
        {"decay": True, "com": False},
        {"decay": True, "com": True},
    ],
    ids=[
        "1->3 decay",
        "1->3 decay, COM",
    ],
)
def mapping_and_args(request):
    com = request.param["com"]
    zeros = np.zeros(N)
    if request.param["decay"]:
        mapping = me.ThreeBodyDecay(com=com)
        if com:

            def make_args(point):
                p0 = np.stack([point.m0, zeros, zeros, zeros], axis=1)
                return (
                    [
                        point.r1,
                        point.r2,
                        point.r3,
                        point.r4,
                        point.r5,
                        point.m0,
                        point.m1,
                        point.m2,
                        point.m3,
                    ],
                    [],
                    p0,
                )

        else:

            def make_args(point):
                return (
                    [
                        point.r1,
                        point.r2,
                        point.r3,
                        point.r4,
                        point.r5,
                        point.m0,
                        point.m1,
                        point.m2,
                        point.m3,
                        point.p0,
                    ],
                    [],
                    point.p0,
                )

    else:
        raise NotImplementedError("Only 1->3 decay is implemented in this test.")
    return (mapping, make_args)


def mass(momentum):
    return np.sqrt(momentum[:, 0] ** 2 - np.sum(momentum[:, 1:] ** 2, axis=1))


InputPoint = namedtuple(
    "InputPoint",
    ["r1", "r2", "r3", "r4", "r5", "m0", "m1", "m2", "m3", "p0", "pa", "pb"],
)

N = 10000


@pytest.fixture
def input_points(rng):
    max_mass = 125.0
    r1 = rng.random(N)
    r2 = rng.random(N)
    r3 = rng.random(N)
    r4 = rng.random(N)
    r5 = rng.random(N)

    pa = np.array([[0, 0, 4000.0]]) + rng.normal(0.0, 500.0, (N, 3))
    ma = rng.uniform(2.5 * max_mass, 1000.0, N)
    ea = np.sqrt(np.sum(pa**2, axis=1) + ma**2)
    pa = np.concatenate([ea[:, None], pa], axis=1)

    pb = np.array([[0, 0, -4000.0]]) + rng.normal(0.0, 500.0, (N, 3))
    mb = rng.uniform(20.0, 500.0, N)
    eb = np.sqrt(np.sum(pb**2, axis=1) + mb**2)
    pb = np.concatenate([eb[:, None], pb], axis=1)

    p0 = pa + pb
    m0 = mass(p0)

    m1 = rng.uniform(1.0, max_mass, N)
    m2 = rng.uniform(1.0, max_mass, N)
    m3 = rng.uniform(1.0, max_mass, N)

    return InputPoint(r1, r2, r3, r4, r5, m0, m1, m2, m3, p0, pa, pb)


def test_momentum_conservation(mapping_and_args, input_points):
    mapping, make_args = mapping_and_args
    inputs, conditions, p0 = make_args(input_points)
    (p1, p2, p3), det = mapping.map_forward(inputs, conditions)
    assert p1 + p2 + p3 == approx(p0)


def test_outgoing_masses(mapping_and_args, input_points):
    mapping, make_args = mapping_and_args
    inputs, conditions, p0 = make_args(input_points)
    (p1, p2, p3), det = mapping.map_forward(inputs, conditions)
    m0 = mass(p1 + p2 + p3)
    m1 = mass(p1)
    m2 = mass(p2)
    m3 = mass(p3)
    index = 5
    print(p3[index])
    print(m3[index], input_points.m3[index])
    assert m0 == approx(input_points.m0)
    assert m1 == approx(input_points.m1)
    assert m2 == approx(input_points.m2)
    assert m3 == approx(input_points.m3)

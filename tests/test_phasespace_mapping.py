import pytest
from pytest import approx
import madevent7 as me
import numpy as np

@pytest.fixture
def rng():
    return np.random.default_rng(1234)

@pytest.fixture(params=[
    [0., 0.],
    [0., 0., 0.],
    [0., 0., 0., 0.],
    [0., 0., 0., 0., 0.],
    [173., 173.],
    [173., 173., 0.],
    [173., 173., 0., 0.],
    [173., 173., 0., 0., 0.],
    [80., 80.],
    [80., 80., 80.],
    [80., 80., 80., 80.],
    [80., 80., 80., 80., 80.],
], ids=[
    "2 particle, massless",
    "3 particle, massless",
    "4 particle, massless",
    "5 particle, massless",
    "2 particle, t tbar",
    "3 particle, t tbar",
    "4 particle, t tbar",
    "5 particle, t tbar",
    "2 particle, W",
    "3 particle, W",
    "4 particle, W",
    "5 particle, W",
])
def masses(request):
    return [0., 0., *request.param]

BATCH_SIZE = 1000
S_LAB = 13000.**2

def test_t_channel_masses(masses, rng):
    mapping = me.PhaseSpaceMapping(masses, S_LAB, mode=me.PhaseSpaceMapping.propagator)
    r = rng.random((BATCH_SIZE, mapping.random_dim()))
    (p_ext, x1, x2), det = mapping.map_forward([r])

    m_ext_true = np.full((BATCH_SIZE, len(masses)), masses)
    m_ext = np.sqrt(np.maximum(0, p_ext[:,:,0]**2 - np.sum(p_ext[:,:,1:]**2, axis=2)))
    assert m_ext == approx(m_ext_true, abs=1e-3, rel=1e-3)

def test_t_channel_incoming(masses, rng):
    mapping = me.PhaseSpaceMapping(masses, S_LAB, mode=me.PhaseSpaceMapping.propagator)
    r = rng.random((BATCH_SIZE, mapping.random_dim()))
    (p_ext, x1, x2), det = mapping.map_forward([r])
    zeros = np.zeros(BATCH_SIZE)
    p_a = p_ext[:,0]
    p_b = p_ext[:,1]
    e_beam = 0.5 * S_LAB**0.5

    assert p_a[:,0] == approx(p_a[:,3]) and p_b[:,0] == approx(-p_b[:,3])
    assert p_a[:,1] == approx(zeros) and p_a[:,2] == approx(zeros)
    assert p_b[:,1] == approx(zeros) and p_b[:,2] == approx(zeros)
    assert np.all(x1 >= 0) and np.all(x1 <= 1)
    assert np.all(x2 >= 0) and np.all(x2 <= 1)
    assert p_a[:,0] == approx(e_beam * x1)
    assert p_b[:,0] == approx(e_beam * x2)

def test_t_channel_momentum_conservation(masses, rng):
    mapping = me.PhaseSpaceMapping(masses, S_LAB, mode=me.PhaseSpaceMapping.propagator)
    r = rng.random((BATCH_SIZE, mapping.random_dim()))
    (p_ext, x1, x2), det = mapping.map_forward([r])
    p_in = np.sum(p_ext[:, :2], axis=1)
    p_out = np.sum(p_ext[:, 2:], axis=1)

    assert p_out == approx(p_in)

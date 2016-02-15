from __future__ import division

import numpy as np
import pytest
import theanets
import theano

PROBE = np.array([-10, -1, -0.1, 0, 0.1, 1, 10], 'f')


@pytest.mark.parametrize('activation, expected', [
    ('linear', PROBE),
    ('logistic', 1 / (1 + np.exp(-PROBE))),
    ('sigmoid', 1 / (1 + np.exp(-PROBE))),
    ('softmax', np.exp(PROBE) / sum(np.exp(PROBE))),
    ('softplus', np.log1p(np.exp(PROBE))),
    ('relu', np.clip(PROBE, 0, 100)),
    ('rect:max', np.clip(PROBE, 0, 100)),
    ('rect:min', np.clip(PROBE, -100, 1)),
    ('rect:minmax', np.clip(np.clip(PROBE, 0, 100), -100, 1)),
    ('norm:mean', PROBE - PROBE.mean()),
    ('norm:max', PROBE / abs(PROBE).max()),
    ('norm:std', PROBE / PROBE.std()),
    ('norm:z', (PROBE - PROBE.mean()) / PROBE.std()),

    # values based on random initial parameters using seed below
    ('prelu', np.array([
        -0.712390,  -0.0753766, -0.0004450307, 0, 0.1, 1, 10], 'f')),
    ('lgrelu', np.array([
        -0.8612113, -0.1478685, -0.0104537, 0, 0.013451, 0.0532337, 1.350188], 'f')),
    ('maxout:3', np.array([
        8.31461, -0.59702, 1.31311, 1.89274, 0.60570, 1.38634, 6.16415], 'f')),

    # combo burgers
    ('relu+tanh', np.tanh(np.clip(PROBE, 0, 100))),
    ('softplus+norm:z', ((np.log1p(np.exp(PROBE)) -
                          np.log1p(np.exp(PROBE)).mean()) /
                         np.log1p(np.exp(PROBE)).std())),
])
def test_activation(activation, expected):
    layer = theanets.layers.Feedforward(inputs='x', size=7, rng=13)
    f = theanets.activations.build(activation, layer)
    actual = f(theano.shared(PROBE))
    if hasattr(actual, 'eval'):
        actual = actual.eval()
    assert np.allclose(actual, expected)


def test_build():
    a = theanets.layers.Feedforward(
        inputs='x', size=3, activation='relu').activate
    assert callable(a)
    assert a.name == 'relu'
    assert a.params == []


def test_build_composed():
    a = theanets.layers.Feedforward(
        inputs='x', size=3, activation='relu+norm:z').activate
    assert callable(a)
    assert a.name == 'norm:z(relu)', a.name
    assert a.params == []


@pytest.mark.parametrize('activation, expected', [
    ('prelu', ['l.leak']),
    ('lgrelu', ['l.gain', 'l.leak']),
    ('maxout:4', ['l.intercept', 'l.slope']),
])
def test_parameters(activation, expected):
    a = theanets.layers.Feedforward(
        inputs='x', size=3, activation=activation, name='l').activate
    assert sorted(p.name for p in a.params) == expected

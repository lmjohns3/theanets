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
    ('elu', np.array([
        -1.15013397, -0.74292195, -0.0999504, 0, 0.1, 1, 10], 'f')),
    ('prelu', np.array([
        -11.50186157, -1.17528522, -0.10503119, 0, 0.1, 1, 10], 'f')),
    ('lgrelu', np.array([
        -10.52778435, -1.04052365, -0.11633276, 0, 0.10640667,
        1.04642045, 10.21983242], 'f')),
    ('maxout:3', np.array([
        16.60424042, 1.80405843, 1.99347568, 0.3595323, -0.513098,
        2.77195668, 0.61599374], 'f')),

    # combo burgers
    ('relu+tanh', np.tanh(np.clip(PROBE, 0, 100))),
    ('softplus+norm:z', ((np.log1p(np.exp(PROBE)) -
                          np.log1p(np.exp(PROBE)).mean()) /
                         np.log1p(np.exp(PROBE)).std())),
])
def test_activation(activation, expected):
    layer = theanets.layers.Feedforward(inputs='in', size=7, rng=13)
    layer.bind(theanets.Network([3]))
    f = theanets.activations.build(activation, layer)
    actual = f(theano.shared(PROBE))
    if hasattr(actual, 'eval'):
        actual = actual.eval()
    assert np.allclose(actual, expected)


def test_build():
    layer = theanets.layers.Feedforward(inputs='in', size=3, activation='relu')
    layer.bind(theanets.Network([3]))
    a = layer.activate
    assert callable(a)
    assert a.name == 'relu'
    assert a.params == []


def test_build_composed():
    layer = theanets.layers.Feedforward(
        inputs='in', size=3, activation='relu+norm:z')
    layer.bind(theanets.Network([3]))
    a = layer.activate
    assert callable(a)
    assert a.name == 'norm:z(relu)', a.name
    assert a.params == []


@pytest.mark.parametrize('activation, expected', [
    ('prelu', ['l.leak']),
    ('lgrelu', ['l.gain', 'l.leak']),
    ('maxout:4', ['l.intercept', 'l.slope']),
])
def test_parameters(activation, expected):
    layer = theanets.layers.Feedforward(
        inputs='in', size=3, activation=activation, name='l')
    layer.bind(theanets.Network([3, layer]))
    assert sorted(p.name for p in layer.activate.params) == expected

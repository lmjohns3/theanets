import numpy as np
import os
import pytest
import theanets

try:
    from itertools import izip as zip
except ImportError:  # python3
    pass

import util as u


class TestNetwork:
    def test_layer_ints(self):
        model = theanets.Regressor((1, 2, 3))
        assert len(model.layers) == 3

    @pytest.mark.parametrize('layers', [
        (1, (2, 'relu'), 3),
        (1, dict(size=2, activation='relu', form='rnn'), 3),
        (1, 2, dict(size=3, inputs='hid1')),
        (1, 2, dict(size=3, inputs=('in', 'hid1'))),
        (1, 2, (1, 'tied')),
        (1, 2, dict(size=1, form='tied', partner='hid1')),
    ])
    def test_layer_tuples(self, layers):
        model = theanets.Regressor(layers)
        assert len(model.layers) == 3

        assert isinstance(model.layers[0], theanets.layers.Input)
        assert model.layers[0].kwargs['activation'] == 'linear'
        assert model.layers[0].output_shape == (1, )

        assert model.layers[1].kwargs['activation'] == 'relu'
        spec = layers[1]
        if isinstance(spec, dict) and spec.get('form') == 'rnn':
            assert isinstance(model.layers[1], theanets.layers.RNN)
        else:
            assert isinstance(model.layers[1], theanets.layers.Feedforward)

        assert model.layers[2].kwargs['activation'] == 'linear'
        spec = layers[2]
        if (isinstance(spec, tuple) and 'tied' in spec) or \
           (isinstance(spec, dict) and spec.get('form') == 'tied'):
            assert isinstance(model.layers[2], theanets.layers.Tied)
            assert model.layers[2].partner is model.layers[1]

    @pytest.mark.parametrize('layers', [
        (1, 2, dict(size=3, inputs='hid2')),
        (1, (2, 'tied'), (2, 'tied'), (1, 'tied')),
    ])
    def test_layers_raises(self, layers):
        with pytest.raises(theanets.util.ConfigurationError):
            theanets.Regressor(layers)

    @pytest.mark.parametrize('spec, cls, shape, act', [
        (6, theanets.layers.Feedforward, (6, ), None),
        ((6, ), theanets.layers.Feedforward, (6, ), None),
        ((6, 7), theanets.layers.Feedforward, (6, 7), None),
        ((6, 'linear'), theanets.layers.Feedforward, (6, ), 'linear'),
        ((6, 'linear', 'classifier'), theanets.layers.Classifier, (6, ), 'softmax'),
        (dict(size=6), theanets.layers.Feedforward, (6, ), None),
        (dict(size=6, form='ff'), theanets.layers.Feedforward, (6, ), None),
        (dict(size=6, activation='linear'), theanets.layers.Feedforward, (6, ), 'linear'),
        (dict(shape=(6, 7)), theanets.layers.Feedforward, (6, 7), None),
    ])
    def test_add_layer(self, spec, cls, shape, act):
        model = theanets.Regressor([3, spec, 4])
        layer = model.layers[1]
        assert len(model.layers) == 3
        assert isinstance(layer, cls)
        assert layer.output_shape == shape
        if act is not None:
            assert layer.kwargs['activation'] == act

    @pytest.mark.parametrize('spec', [
        (6, 'tied', 7),
        None,
        'ff',
        'tied',
        dict(form='ff'),
        dict(form='tied'),
        dict(form='tied', partner='hello'),
        dict(form='ff', inputs=('a', 'b')),
    ])
    def test_add_layer_errors(self, spec):
        with pytest.raises(theanets.util.ConfigurationError):
            theanets.Network([dict(form='input', name='a', shape=(3, 5)),
                              dict(form='input', name='b', shape=(4, 3)),
                              spec,
                              4])

    def test_updates(self):
        model = theanets.Regressor((15, 13))
        assert not model.updates()

    def test_default_output_name(self):
        model = theanets.Regressor((1, 2, dict(size=1, form='tied', name='foo')))
        assert model.losses[0].output_name == 'foo:out'
        model = theanets.Regressor((1, 2, 1))
        assert model.losses[0].output_name == 'out:out'

    def test_find_number(self):
        model = theanets.Regressor((1, 2, 1))
        p = model.find(1, 0)
        assert p.name == 'hid1.w'
        p = model.find(2, 0)
        assert p.name == 'out.w'

    def test_find_name(self):
        model = theanets.Regressor((1, 2, 1))
        p = model.find('hid1', 'w')
        assert p.name == 'hid1.w'
        p = model.find('out', 'w')
        assert p.name == 'out.w'

    def test_find_missing(self):
        model = theanets.Regressor((1, 2, 1))
        try:
            model.find('hid4', 'w')
            assert False
        except KeyError:
            pass
        try:
            model.find(0, 0)
            assert False
        except KeyError:
            pass
        try:
            model.find(1, 3)
            assert False
        except KeyError:
            pass

    def test_train(self):
        model = theanets.Regressor((1, 2, 1))
        tm, vm = model.train([np.random.randn(100, 1).astype('f'),
                              np.random.randn(100, 1).astype('f')])
        assert tm['loss'] > 0


class TestMonitors:
    @pytest.fixture
    def net(self):
        return theanets.Regressor((10, 15, 14, 13))

    def assert_monitors(self, net, monitors, expected, sort=False):
        mon = [k for k, v in net.monitors(monitors=monitors)]
        if sort:
            mon = sorted(mon)
        assert mon == expected

    def test_dict(self, net):
        self.assert_monitors(net, {'hid1:out': 1}, ['err', 'hid1:out<1'])

    def test_list(self, net):
        self.assert_monitors(net, [('hid1:out', 1)], ['err', 'hid1:out<1'])

    def test_list_values(self, net):
        self.assert_monitors(
            net, {'hid1:out': [2, 1]}, ['err', 'hid1:out<2', 'hid1:out<1'])

    def test_dict_values(self, net):
        self.assert_monitors(
            net, {'hid1:out': dict(a=lambda e: e+1, b=lambda e: e+2)},
            ['err', 'hid1:out:a', 'hid1:out:b'], sort=True)

    def test_not_found(self, net):
        self.assert_monitors(net, {'hid10:out': 1}, ['err'])

    def test_param(self, net):
        self.assert_monitors(net, {'hid1.w': 1}, ['err', 'hid1.w<1'])

    def test_wildcard(self, net):
        self.assert_monitors(
            net, {'*.w': 1}, ['err', 'hid1.w<1', 'hid2.w<1', 'out.w<1'])
        self.assert_monitors(net, {'hid?.w': 1}, ['err', 'hid1.w<1', 'hid2.w<1'])


def test_save_every(tmpdir):
    net = theanets.Autoencoder((u.NUM_INPUTS, (3, 'prelu'), u.NUM_INPUTS))
    p = tmpdir.mkdir('graph-test').join('model.pkl')
    fn = os.path.join(p.dirname, p.basename)
    train = net.itertrain([u.INPUTS], save_every=2, save_progress=fn)
    for i, _ in enumerate(zip(train, range(9))):
        if i == 3 or i == 5 or i == 7:
            assert p.check()
        else:
            assert not p.check()
        if p.check():
            p.remove()

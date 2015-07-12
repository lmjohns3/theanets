import numpy as np
import os
import tempfile
import theanets

from itertools import izip as zip

import util


class TestNetwork:
    def test_updates(self):
        m = theanets.Regressor((15, 13))
        assert not m.updates()

    def test_layer_ints(self):
        m = theanets.Regressor((1, 2, 3))
        assert len(m.layers) == 3

    def test_layer_tuples(self):
        m = theanets.Regressor((1, (2, 'relu'), 3))
        assert len(m.layers) == 3
        assert m.layers[1].activation == 'relu'

    def test_layer_dicts(self):
        m = theanets.Regressor((1, dict(size=2, activation='relu', form='rnn'), 3))
        assert len(m.layers) == 3
        assert m.layers[1].activation == 'relu'
        assert isinstance(m.layers[1], theanets.layers.recurrent.RNN)

    def test_layer_tied(self):
        m = theanets.Regressor((1, 2, (1, 'tied')))
        assert len(m.layers) == 3
        assert isinstance(m.layers[2], theanets.layers.feedforward.Tied)
        assert m.layers[2].partner is m.layers[1]

    def test_find_number(self):
        m = theanets.Regressor((1, 2, 1))
        p = m.find(1, 0)
        assert p.name == 'hid1.w'
        p = m.find(2, 0)
        assert p.name == 'out.w'

    def test_find_name(self):
        m = theanets.Regressor((1, 2, 1))
        p = m.find('hid1', 'w')
        assert p.name == 'hid1.w'
        p = m.find('out', 'w')
        assert p.name == 'out.w'

    def test_find_missing(self):
        m = theanets.Regressor((1, 2, 1))
        try:
            m.find('hid4', 'w')
            assert False
        except KeyError:
            pass
        try:
            m.find(0, 0)
            assert False
        except KeyError:
            pass
        try:
            m.find(1, 3)
            assert False
        except KeyError:
            pass


class TestMonitors:
    def setUp(self):
        self.net = theanets.Regressor((10, 15, 14, 13))

    def assert_monitors(self, monitors, expected, sort=False):
        mon = [k for k, v in self.net.monitors(monitors=monitors)]
        if sort:
            mon = sorted(mon)
        assert mon == expected, 'expected {}, got {}'.format(expected, mon)

    def test_dict(self):
        self.assert_monitors({'hid1:out': 1}, ['err', 'hid1:out<1'])

    def test_list(self):
        self.assert_monitors([('hid1:out', 1)], ['err', 'hid1:out<1'])

    def test_list_values(self):
        self.assert_monitors(
            {'hid1:out': [2, 1]}, ['err', 'hid1:out<2', 'hid1:out<1'])

    def test_dict_values(self):
        self.assert_monitors(
            {'hid1:out': dict(a=lambda e: e+1, b=lambda e: e+2)},
            ['err', 'hid1:out:a', 'hid1:out:b'], sort=True)

    def test_not_found(self):
        self.assert_monitors({'hid10:out': 1}, ['err'])

    def test_param(self):
        self.assert_monitors({'hid1.w': 1}, ['err', 'hid1.w<1'])

    def test_wildcard(self):
        self.assert_monitors({'*.w': 1}, ['err', 'hid1.w<1', 'hid2.w<1', 'out.w<1'])
        self.assert_monitors({'hid?.w': 1}, ['err', 'hid1.w<1', 'hid2.w<1'])


class TestSaving(util.Base):
    def test_save_every(self):
        net = theanets.Autoencoder((self.NUM_INPUTS, (3, 'prelu'), self.NUM_INPUTS))
        f, p = tempfile.mkstemp(suffix='pkl')
        os.close(f)
        os.unlink(p)
        train = net.itertrain([self.INPUTS], save_every=2, save_progress=p)
        for i, _ in enumerate(zip(train, range(9))):
            if i == 3 or i == 5 or i == 7:
                assert os.path.isfile(p)
            else:
                assert not os.path.isfile(p)
            if os.path.exists(p):
                os.unlink(p)

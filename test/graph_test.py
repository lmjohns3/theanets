import theanets
import numpy as np

import util


class TestNetwork(util.MNIST):
    def _build(self, *hiddens):
        return theanets.Regressor((self.DIGIT_SIZE, ) + hiddens)

    def test_transform(self):
        net = self._build(15, 13)
        y = net.transform(self.images)
        assert y.shape == (self.NUM_DIGITS, 13)

    def test_feed_forward(self):
        net = self._build(15, 13)
        hs = net.feed_forward(self.images)
        assert len(hs) == 7, 'got {}'.format(list(hs.keys()))
        assert hs['in:out'].shape == (self.NUM_DIGITS, self.DIGIT_SIZE)
        assert hs['hid1:out'].shape == (self.NUM_DIGITS, 15)
        assert hs['out:out'].shape == (self.NUM_DIGITS, 13)

    def test_decode_from_multiple_layers(self):
        net = self._build(13, 14, dict(
            size=15, inputs={'hid2:out': 14, 'hid1:out': 13}))
        hs = net.feed_forward(self.images)
        assert len(hs) == 9, 'got {}'.format(list(hs.keys()))
        assert hs['in:out'].shape == (self.NUM_DIGITS, self.DIGIT_SIZE)
        assert hs['hid1:out'].shape == (self.NUM_DIGITS, 13)
        assert hs['hid2:out'].shape == (self.NUM_DIGITS, 14)
        assert hs['out:out'].shape == (self.NUM_DIGITS, 15)

    def test_updates(self):
        assert not self._build(13).updates()


class TestMonitors(TestNetwork):
    def setUp(self):
        super(TestMonitors, self).setUp()
        self.net = self._build(15, 13)

    def assert_monitors(self, monitors, expected):
        mon = [k for k, v in self.net.monitors(monitors=monitors)]
        assert mon == expected, 'expected {}, got {}'.format(expected, mon)

    def test_dict(self):
        self.assert_monitors({'hid1:out': 1}, ['err', 'hid1:out<1'])

    def test_list(self):
        self.assert_monitors([('hid1:out', 1)], ['err', 'hid1:out<1'])

    def test_list_values(self):
        self.assert_monitors({'hid1:out': [2, 1]},
                             ['err', 'hid1:out<2', 'hid1:out<1'])

    def test_dict_values(self):
        self.assert_monitors({'hid1:out': dict(a=lambda e: e+1,
                                               b=lambda e: e+2)},
                             ['err', 'hid1:out:b', 'hid1:out:a'])

    def test_not_found(self):
        self.assert_monitors({'hid10:out': 1}, ['err'])

    def test_param(self):
        self.assert_monitors({'hid1.w': 1}, ['err', 'hid1.w<1'])

    def test_wildcard(self):
        self.assert_monitors({'*.w': 1}, ['err', 'hid1.w<1', 'out.w<1'])
        self.assert_monitors({'hid?.w': 1}, ['err', 'hid1.w<1'])

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

    def test_monitor_dict(self):
        net = self._build(15, 13)
        mon = net.monitors(monitors={'hid1:out': 1})
        assert mon == (('hid1:out<1', )), 'got {}'.format(mon)

    def test_monitor_callable(self):
        net = self._build(15, 13)
        assert net.monitors({'hid1:out': lambda e: e + 1}) == ()

    def test_monitor_list(self):
        net = self._build(15, 13)
        net.monitors([])

    def test_monitor_values(self):
        net = self._build(15, 13)
        net.monitors([])

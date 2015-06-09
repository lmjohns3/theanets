import numpy as np
import theanets

import util


class TestRegressor(util.MNIST):
    def _build(self, *hiddens):
        return theanets.Regressor((self.DIGIT_SIZE, ) + hiddens + (3, ))

    def test_predict(self):
        net = self._build(15, 13)
        y = net.predict(self.images)
        assert y.shape == (self.NUM_DIGITS, 3)

    def test_score_onelayer(self):
        net = self._build(13)
        z = net.score(self.images)
        assert z < 0

    def test_feed_forward(self):
        net = self._build(15, 13)
        hs = net.feed_forward(self.images)
        assert len(hs) == 9, 'got {}'.format(list(hs.keys()))
        assert hs['in:out'].shape == (self.NUM_DIGITS, self.DIGIT_SIZE)
        assert hs['hid1:out'].shape == (self.NUM_DIGITS, 15)
        assert hs['out:out'].shape == (self.NUM_DIGITS, 3)

    def test_decode_from_multiple_layers(self):
        net = self._build(13, 14, dict(
            size=15, inputs={'hid2:out': 14, 'hid1:out': 13}))
        hs = net.feed_forward(self.images)
        assert len(hs) == 11, 'got {}'.format(list(hs.keys()))
        assert hs['in:out'].shape == (self.NUM_DIGITS, self.DIGIT_SIZE)
        assert hs['hid1:out'].shape == (self.NUM_DIGITS, 13)
        assert hs['hid2:out'].shape == (self.NUM_DIGITS, 14)
        assert hs['out:out'].shape == (self.NUM_DIGITS, 3)


class TestWeightedRegressor(TestRegressor):
    def _build(self, *hiddens):
        return theanets.Regressor((self.DIGIT_SIZE, ) + hiddens + (3, ), weighted=True)


class TestClassifier(util.MNIST):
    def _build(self, *hiddens):
        return theanets.Classifier((self.DIGIT_SIZE, ) + hiddens + (10, ))

    def test_predict_onelayer(self):
        net = self._build(13)
        z = net.predict(self.images)
        assert z.shape == (self.NUM_DIGITS, )

    def test_score_onelayer(self):
        net = self._build(13)
        z = net.score(self.images, self.labels)
        assert 0 < z < 1

    def test_predict_proba_onelayer(self):
        net = self._build(13)
        z = net.predict_proba(self.images)
        assert z.shape == (self.NUM_DIGITS, 10)

    def test_predict_twolayer(self):
        net = self._build(13, 14)
        z = net.predict(self.images)
        assert z.shape == (self.NUM_DIGITS, )


class TestAutoencoder(util.MNIST):
    def _build(self, *hiddens):
        return theanets.Autoencoder(
            (self.DIGIT_SIZE, ) + hiddens + (self.DIGIT_SIZE, ))

    def test_encode_onelayer(self):
        net = self._build(13)
        z = net.encode(self.images)
        assert z.shape == (self.NUM_DIGITS, 13)

    def test_encode_twolayer(self):
        net = self._build(13, 14)
        z = net.encode(self.images)
        assert z.shape == (self.NUM_DIGITS, 14)

    def test_encode_threelayer(self):
        net = self._build(13, 14, 15)
        z = net.encode(self.images)
        assert z.shape == (self.NUM_DIGITS, 14)

    def test_decode_onelayer(self):
        net = self._build(13)
        x = net.decode(net.encode(self.images))
        assert x.shape == (self.NUM_DIGITS, self.DIGIT_SIZE)

    def test_decode_twolayer(self):
        net = self._build(13, 14)
        x = net.decode(net.encode(self.images))
        assert x.shape == (self.NUM_DIGITS, self.DIGIT_SIZE)

    def test_decode_threelayer(self):
        net = self._build(13, 14, 15)
        x = net.decode(net.encode(self.images))
        assert x.shape == (self.NUM_DIGITS, self.DIGIT_SIZE)

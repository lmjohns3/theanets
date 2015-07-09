import numpy as np
import scipy.sparse
import theanets

import util


class Base(util.Base):
    INPUTS = scipy.sparse.csr_matrix(util.Base.INPUTS)


class TestRegressor(Base):
    def _build(self, *hiddens):
        return theanets.Regressor(
            [self.NUM_INPUTS] + list(hiddens) + [self.NUM_OUTPUTS],
            sparse_input=True)

    def test_sgd(self):
        self.exp = theanets.Experiment(
            theanets.Regressor,
            layers=[self.NUM_INPUTS, 10, self.NUM_OUTPUTS],
            sparse_input=True)
        self.assert_progress('sgd', [self.INPUTS, self.OUTPUTS])

    def test_predict(self):
        net = self._build(15, 13)
        y = net.predict(self.INPUTS)
        self.assert_shape(y.shape, self.NUM_OUTPUTS)

    def test_score_onelayer(self):
        net = self._build(13)
        z = net.score(self.INPUTS, self.OUTPUTS)
        assert z < 0

    def test_feed_forward(self):
        net = self._build(15, 13)
        hs = net.feed_forward(self.INPUTS)
        assert len(hs) == 9, 'got {}'.format(list(hs.keys()))
        self.assert_shape(hs['in:out'].shape, self.NUM_INPUTS)
        self.assert_shape(hs['hid1:out'].shape, 15)
        self.assert_shape(hs['out:out'].shape, self.NUM_OUTPUTS)

    def test_decode_from_multiple_layers(self):
        net = self._build(13, 14, dict(
            size=15, inputs={'hid2:out': 14, 'hid1:out': 13}))
        hs = net.feed_forward(self.INPUTS)
        assert len(hs) == 11, 'got {}'.format(list(hs.keys()))
        self.assert_shape(hs['in:out'].shape, self.NUM_INPUTS)
        self.assert_shape(hs['hid1:out'].shape, 13)
        self.assert_shape(hs['hid2:out'].shape, 14)
        self.assert_shape(hs['out:out'].shape, self.NUM_OUTPUTS)


class TestClassifier(Base):
    def _build(self, *hiddens):
        return theanets.Classifier(
            [self.NUM_INPUTS] + list(hiddens) + [self.NUM_CLASSES],
            sparse_input=True)

    def test_sgd(self):
        self.exp = theanets.Experiment(
            theanets.Classifier,
            layers=(self.NUM_INPUTS, 10, self.NUM_CLASSES),
            sparse_input=True)
        self.assert_progress('sgd', [self.INPUTS, self.CLASSES])

    def test_predict_onelayer(self):
        net = self._build(13)
        z = net.predict(self.INPUTS)
        self.assert_shape(z.shape, (self.NUM_EXAMPLES, ))

    def test_score_onelayer(self):
        net = self._build(13)
        z = net.score(self.INPUTS, self.CLASSES)
        assert 0 <= z <= 1

    def test_predict_proba_onelayer(self):
        net = self._build(13)
        z = net.predict_proba(self.INPUTS)
        self.assert_shape(z.shape, self.NUM_CLASSES)

    def test_predict_twolayer(self):
        net = self._build(13, 14)
        z = net.predict(self.INPUTS)
        self.assert_shape(z.shape, (self.NUM_EXAMPLES, ))

    def test_feed_forward(self):
        net = self._build(15, 13)
        hs = net.feed_forward(self.INPUTS)
        assert len(hs) == 9, 'got {}'.format(list(hs.keys()))
        self.assert_shape(hs['in:out'].shape, self.NUM_INPUTS)
        self.assert_shape(hs['hid1:out'].shape, 15)
        self.assert_shape(hs['out:out'].shape, self.NUM_CLASSES)

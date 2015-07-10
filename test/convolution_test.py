import numpy as np
import theanets

import util


class Base(util.Base):
    NUM_WIDTH = 13
    NUM_HEIGHT = 15
    NUM_EXAMPLES = util.Base.NUM_EXAMPLES
    NUM_INPUTS = util.Base.NUM_INPUTS
    NUM_OUTPUTS = util.Base.NUM_OUTPUTS
    NUM_CLASSES = util.Base.NUM_CLASSES

    INPUTS = np.random.randn(
        NUM_EXAMPLES, NUM_WIDTH, NUM_HEIGHT, NUM_INPUTS).astype('f')
    INPUT_WEIGHTS = abs(np.random.randn(
        NUM_EXAMPLES, NUM_WIDTH, NUM_HEIGHT, NUM_INPUTS)).astype('f')

    def assert_shape(self, actual, expected):
        if not isinstance(expected, tuple):
            expected = (self.NUM_EXAMPLES, self.NUM_WIDTH, self.NUM_HEIGHT, expected)
        assert actual == expected, 'expected {}, got {}'.format(expected, actual)


class TestRegressor(Base):
    def _build(self, *hiddens):
        return theanets.convolution.Regressor(
            [self.NUM_INPUTS] + list(hiddens) + [self.NUM_OUTPUTS])

    def test_sgd(self):
        self.exp = theanets.Experiment(
            theanets.convolution.Regressor,
            layers=(self.NUM_INPUTS, (10, 'conv2'), self.NUM_OUTPUTS))
        self.assert_progress('sgd', [self.INPUTS, self.OUTPUTS])

    def test_predict(self):
        net = self._build(15, 13)
        y = net.predict(self.INPUTS)
        self.assert_shape(y.shape, self.NUM_OUTPUTS)

    def test_feed_forward(self):
        net = self._build(15, 13)
        hs = net.feed_forward(self.INPUTS)
        assert len(hs) == 9, 'got {}'.format(list(hs.keys()))
        self.assert_shape(hs['in:out'].shape, self.NUM_INPUTS)
        self.assert_shape(hs['hid1:out'].shape, 15)
        self.assert_shape(hs['hid2:out'].shape, 13)
        self.assert_shape(hs['out:out'].shape, self.NUM_OUTPUTS)

    def test_multiple_recurrent(self):
        net = self._build(13, 14, 15)
        hs = net.feed_forward(self.INPUTS)
        assert len(hs) == 11, 'got {}'.format(list(hs.keys()))
        self.assert_shape(hs['in:out'].shape, self.NUM_INPUTS)
        self.assert_shape(hs['hid1:out'].shape, 13)
        self.assert_shape(hs['hid2:out'].shape, 14)
        self.assert_shape(hs['hid3:out'].shape, 15)
        self.assert_shape(hs['out:out'].shape, self.NUM_OUTPUTS)


class TestWeightedRegressor(TestRegressor):
    def _build(self, *hiddens):
        return theanets.convolution.Regressor(
            [self.NUM_INPUTS] + list(hiddens) + [self.NUM_OUTPUTS],
            weighted=True)

    def test_sgd(self):
        self.exp = theanets.Experiment(
            theanets.convolution.Regressor,
            layers=(self.NUM_INPUTS, (10, 'conv2'), self.NUM_OUTPUTS),
            weighted=True)
        self.assert_progress('sgd', [self.INPUTS, self.OUTPUTS, self.OUTPUT_WEIGHTS])


class TestClassifier(Base):
    def _build(self, *hiddens):
        return theanets.convolution.Classifier(
            [self.NUM_INPUTS] + list(hiddens) + [self.NUM_OUTPUTS])

    def test_sgd(self):
        self.exp = theanets.Experiment(
            theanets.convolution.Classifier,
            layers=(self.NUM_INPUTS, (10, 'conv2'), self.NUM_CLASSES))
        self.assert_progress('sgd', [self.INPUTS, self.CLASSES])

    def test_predict_onelayer(self):
        net = self._build(13)
        z = net.predict(self.INPUTS)
        self.assert_shape(z.shape, (self.NUM_EXAMPLES, ))

    def test_score_onelayer(self):
        net = self._build(13)
        z = net.score(self.INPUTS, self.CLASSES)
        assert 0 < z < 1

    def test_predict_proba_onelayer(self):
        net = self._build(13)
        z = net.predict_proba(self.INPUTS)
        self.assert_shape(z.shape, self.NUM_OUTPUTS)

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
        self.assert_shape(hs['hid2:out'].shape, 13)
        self.assert_shape(hs['out:out'].shape, self.NUM_OUTPUTS)

    def test_predict_sequence(self):
        net = self._build(13)

        count = 0
        for cs in net.predict_sequence([0, 0, 1, 2], 4, streams=3):
            assert isinstance(cs, list)
            assert len(cs) == 3
            count += 1
        assert count == 4

        count = 0
        for cs in net.predict_sequence([0, 0, 1, 2], 4):
            print(cs, type(cs))
            assert isinstance(cs, int)
            count += 1
        assert count == 4


class TestWeightedClassifier(TestClassifier):
    def _build(self, *hiddens):
        return theanets.convolution.Classifier(
            [self.NUM_INPUTS] + list(hiddens) + [self.NUM_OUTPUTS],
            weighted=True)

    def test_sgd(self):
        self.exp = theanets.Experiment(
            theanets.convolution.Classifier,
            layers=(self.NUM_INPUTS, (10, 'conv2'), self.NUM_CLASSES),
            weighted=True)
        self.assert_progress('sgd', [self.INPUTS, self.CLASSES, self.CLASS_WEIGHTS])

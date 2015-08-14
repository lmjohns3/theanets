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

    FILTER_WIDTH = 4
    FILTER_HEIGHT = 3
    FILTER_SIZE = (FILTER_WIDTH, FILTER_HEIGHT)

    INPUTS = np.random.randn(
        NUM_EXAMPLES, NUM_WIDTH, NUM_HEIGHT, NUM_INPUTS).astype('f')
    INPUT_WEIGHTS = abs(np.random.randn(
        NUM_EXAMPLES, NUM_WIDTH, NUM_HEIGHT, NUM_INPUTS)).astype('f')

    def assert_shape(self, actual, expected):
        if isinstance(expected, int):
            expected = (expected, )
        expected = (self.NUM_EXAMPLES, ) + tuple(expected)
        assert actual == expected, 'expected {}, got {}'.format(expected, actual)


class TestRegressor(Base):
    def _build(self, *hiddens):
        hid = [dict(form='conv2', size=h, filter_size=self.FILTER_SIZE) for h in hiddens]
        width = self.NUM_WIDTH - (self.FILTER_WIDTH - 1) * len(hiddens)
        height = self.NUM_HEIGHT - (self.FILTER_HEIGHT - 1) * len(hiddens)
        out = [('flatten', width * height * hiddens[-1]), self.NUM_OUTPUTS]
        return theanets.convolution.Regressor([self.NUM_INPUTS] + hid + out)

    def test_sgd(self):
        hid = dict(size=10, form='conv2', filter_size=self.FILTER_SIZE)
        w = self.NUM_WIDTH - (self.FILTER_WIDTH - 1)
        h = self.NUM_HEIGHT - (self.FILTER_HEIGHT - 1)
        flat = ('flatten', 10 * w * h)
        self.exp = theanets.Experiment(
            theanets.convolution.Regressor,
            layers=(self.NUM_INPUTS, hid, flat, self.NUM_OUTPUTS))
        self.assert_progress('sgd', [self.INPUTS, self.OUTPUTS])

    def test_predict(self):
        net = self._build(15, 13)
        y = net.predict(self.INPUTS)
        self.assert_shape(y.shape, self.NUM_OUTPUTS)

    def test_feed_forward(self):
        net = self._build(15, 13)
        hs = net.feed_forward(self.INPUTS)
        assert len(hs) == 10, 'got {}'.format(list(hs.keys()))
        w = self.NUM_WIDTH
        h = self.NUM_HEIGHT
        self.assert_shape(hs['in:out'].shape, (w, h, self.NUM_INPUTS))
        w -= self.FILTER_WIDTH - 1
        h -= self.FILTER_HEIGHT - 1
        self.assert_shape(hs['hid1:out'].shape, (w, h, 15))
        w -= self.FILTER_WIDTH - 1
        h -= self.FILTER_HEIGHT - 1
        self.assert_shape(hs['hid2:out'].shape, (w, h, 13))
        self.assert_shape(hs['hid3:out'].shape, w * h * 13)
        self.assert_shape(hs['out:out'].shape, self.NUM_OUTPUTS)


class TestWeightedRegressor(TestRegressor):
    def _build(self, *hiddens):
        hid = [dict(form='conv2', size=h, filter_size=self.FILTER_SIZE) for h in hiddens]
        width = self.NUM_WIDTH - (self.FILTER_WIDTH - 1) * len(hiddens)
        height = self.NUM_HEIGHT - (self.FILTER_HEIGHT - 1) * len(hiddens)
        out = [('flatten', width * height * hiddens[-1]), self.NUM_OUTPUTS]
        return theanets.convolution.Regressor(
            [self.NUM_INPUTS] + hid + out, weighted=True)

    def test_sgd(self):
        hid = dict(size=10, form='conv2', filter_size=self.FILTER_SIZE)
        w = self.NUM_WIDTH - (self.FILTER_WIDTH - 1)
        h = self.NUM_HEIGHT - (self.FILTER_HEIGHT - 1)
        flat = ('flatten', 10 * w * h)
        self.exp = theanets.Experiment(
            theanets.convolution.Regressor,
            layers=(self.NUM_INPUTS, hid, flat, self.NUM_OUTPUTS),
            weighted=True)
        self.assert_progress('sgd', [self.INPUTS, self.OUTPUTS, self.OUTPUT_WEIGHTS])


class TestClassifier(Base):
    def _build(self, *hiddens):
        hid = [dict(form='conv2', size=h, filter_size=self.FILTER_SIZE) for h in hiddens]
        width = self.NUM_WIDTH - (self.FILTER_WIDTH - 1) * len(hiddens)
        height = self.NUM_HEIGHT - (self.FILTER_HEIGHT - 1) * len(hiddens)
        out = [('flatten', width * height * hiddens[-1]), self.NUM_CLASSES]
        return theanets.convolution.Classifier([self.NUM_INPUTS] + hid + out)

    def test_sgd(self):
        hid = dict(size=10, form='conv2', filter_size=self.FILTER_SIZE)
        w = self.NUM_WIDTH - (self.FILTER_WIDTH - 1)
        h = self.NUM_HEIGHT - (self.FILTER_HEIGHT - 1)
        flat = ('flatten', 10 * w * h)
        self.exp = theanets.Experiment(
            theanets.convolution.Classifier,
            layers=(self.NUM_INPUTS, hid, flat, self.NUM_CLASSES))
        self.assert_progress('sgd', [self.INPUTS, self.CLASSES])

    def test_predict_onelayer(self):
        net = self._build(13)
        z = net.predict(self.INPUTS)
        self.assert_shape(z.shape, ())

    def test_score_onelayer(self):
        net = self._build(13)
        z = net.score(self.INPUTS, self.CLASSES)
        assert 0 < z < 1

    def test_predict_proba_onelayer(self):
        net = self._build(13)
        z = net.predict_proba(self.INPUTS)
        self.assert_shape(z.shape, self.NUM_CLASSES)

    def test_predict_twolayer(self):
        net = self._build(13, 14)
        z = net.predict(self.INPUTS)
        self.assert_shape(z.shape, ())

    def test_feed_forward(self):
        net = self._build(15, 13)
        hs = net.feed_forward(self.INPUTS)
        assert len(hs) == 10, 'got {}'.format(list(hs.keys()))
        w = self.NUM_WIDTH
        h = self.NUM_HEIGHT
        self.assert_shape(hs['in:out'].shape, (w, h, self.NUM_INPUTS))
        w -= self.FILTER_WIDTH - 1
        h -= self.FILTER_HEIGHT - 1
        self.assert_shape(hs['hid1:out'].shape, (w, h, 15))
        w -= self.FILTER_WIDTH - 1
        h -= self.FILTER_HEIGHT - 1
        self.assert_shape(hs['hid2:out'].shape, (w, h, 13))
        self.assert_shape(hs['hid3:out'].shape, w * h * 13)
        self.assert_shape(hs['out:out'].shape, self.NUM_CLASSES)


class TestWeightedClassifier(TestClassifier):
    def _build(self, *hiddens):
        hid = [dict(form='conv2', size=h, filter_size=self.FILTER_SIZE) for h in hiddens]
        width = self.NUM_WIDTH - (self.FILTER_WIDTH - 1) * len(hiddens)
        height = self.NUM_HEIGHT - (self.FILTER_HEIGHT - 1) * len(hiddens)
        out = [('flatten', width * height * hiddens[-1]), self.NUM_CLASSES]
        return theanets.convolution.Classifier(
            [self.NUM_INPUTS] + hid + out, weighted=True)

    def test_sgd(self):
        hid = dict(size=10, form='conv2', filter_size=self.FILTER_SIZE)
        w = self.NUM_WIDTH - (self.FILTER_WIDTH - 1)
        h = self.NUM_HEIGHT - (self.FILTER_HEIGHT - 1)
        flat = ('flatten', 10 * w * h)
        self.exp = theanets.Experiment(
            theanets.convolution.Classifier,
            layers=(self.NUM_INPUTS, hid, flat, self.NUM_CLASSES),
            weighted=True)
        self.assert_progress('sgd', [self.INPUTS, self.CLASSES, self.CLASS_WEIGHTS])

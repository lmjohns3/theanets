import theanets
import numpy as np

INS = 3
OUTS = 2
STEPS = 11
ALL = 30
BATCH = 10


class TestFunctions:
    def setUp(self):
        self.samples = np.random.randn(2 * STEPS, INS)
        self.labels = np.random.randn(2 * STEPS, OUTS)

    def test_batches_labeled(self):
        f = theanets.recurrent.batches(
            self.samples, self.labels, steps=STEPS, batch_size=BATCH)
        assert len(f()) == 2
        assert f()[0].shape == (STEPS, BATCH, INS)
        assert f()[1].shape == (STEPS, BATCH, OUTS)

    def test_batches_unlabeled(self):
        f = theanets.recurrent.batches(
            self.samples, steps=STEPS, batch_size=BATCH)
        assert len(f()) == 1
        assert f()[0].shape == (STEPS, BATCH, INS)


class Base:
    def setUp(self):
        np.random.seed(3)
        self.inputs = np.random.randn(STEPS, ALL, INS).astype('f')
        self.outputs = np.random.randn(STEPS, ALL, OUTS).astype('f')
        self.probe = np.random.randn(STEPS, BATCH, INS).astype('f')

    def assert_shape(self, actual, expected):
        assert actual == expected, 'expected {}, got {}'.format(expected, actual)


class TestNetwork(Base):
    def _build(self, *hiddens):
        return theanets.recurrent.Regressor(layers=(INS, ) + hiddens + (OUTS, ))

    def test_predict(self):
        net = self._build(15, 13)
        y = net.predict(self.probe)
        self.assert_shape(y.shape, (STEPS, BATCH, OUTS))

    def test_feed_forward(self):
        net = self._build(15, 13)
        hs = net.feed_forward(self.probe)
        assert len(hs) == 4
        self.assert_shape(hs[0].shape, (STEPS, BATCH, INS))
        self.assert_shape(hs[1].shape, (STEPS, BATCH, 15))
        self.assert_shape(hs[2].shape, (STEPS, BATCH, 13))
        self.assert_shape(hs[3].shape, (STEPS, BATCH, OUTS))

    def test_multiple_recurrent(self):
        net = self._build(13, 14, 15)
        hs = net.feed_forward(self.probe)
        assert len(hs) == 5
        self.assert_shape(hs[0].shape, (STEPS, BATCH, INS))
        self.assert_shape(hs[1].shape, (STEPS, BATCH, 13))
        self.assert_shape(hs[2].shape, (STEPS, BATCH, 14))
        self.assert_shape(hs[3].shape, (STEPS, BATCH, 15))
        self.assert_shape(hs[4].shape, (STEPS, BATCH, OUTS))


class TestPredictor(Base):
    def _build(self, *hiddens):
        return theanets.recurrent.Predictor((INS, ) + hiddens + (INS, ))

    def test_predict_onelayer(self):
        net = self._build(13)
        z = net.predict(self.probe)
        self.assert_shape(z.shape, (STEPS, BATCH, INS))


class TestClassifier(Base):
    def _build(self, *hiddens):
        return theanets.recurrent.Classifier((INS, ) + hiddens + (OUTS, ))

    def test_classify_onelayer(self):
        net = self._build(13)
        z = net.classify(self.probe)
        self.assert_shape(z.shape, (STEPS, BATCH))

    def test_classify_twolayer(self):
        net = self._build(13, 14)
        z = net.classify(self.probe)
        self.assert_shape(z.shape, (STEPS, BATCH))


class TestAutoencoder(Base):
    def _build(self, *hiddens):
        return theanets.recurrent.Autoencoder((INS, ) + hiddens + (INS, ))

    def test_encode_onelayer(self):
        net = self._build(13)
        z = net.predict(self.probe)
        self.assert_shape(z.shape, (STEPS, BATCH, INS))

    def test_encode_twolayer(self):
        net = self._build(13, 14)
        z = net.predict(self.probe)
        self.assert_shape(z.shape, (STEPS, BATCH, INS))

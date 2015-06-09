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
        self.probe_classes = np.random.randn(STEPS, BATCH).astype('i')

    def assert_shape(self, actual, expected):
        assert actual == expected, 'expected {}, got {}'.format(expected, actual)


class TestRegressor(Base):
    def _build(self, *hiddens):
        return theanets.recurrent.Regressor(layers=(INS, ) + hiddens + (OUTS, ))

    def test_predict(self):
        net = self._build(15, 13)
        y = net.predict(self.probe)
        self.assert_shape(y.shape, (STEPS, BATCH, OUTS))

    def test_feed_forward(self):
        net = self._build(15, 13)
        hs = net.feed_forward(self.probe)
        assert len(hs) == 9, 'got {}'.format(list(hs.keys()))
        self.assert_shape(hs['in:out'].shape, (STEPS, BATCH, INS))
        self.assert_shape(hs['hid1:out'].shape, (STEPS, BATCH, 15))
        self.assert_shape(hs['hid2:out'].shape, (STEPS, BATCH, 13))
        self.assert_shape(hs['out:out'].shape, (STEPS, BATCH, OUTS))

    def test_multiple_recurrent(self):
        net = self._build(13, 14, 15)
        hs = net.feed_forward(self.probe)
        assert len(hs) == 11, 'got {}'.format(list(hs.keys()))
        self.assert_shape(hs['in:out'].shape, (STEPS, BATCH, INS))
        self.assert_shape(hs['hid1:out'].shape, (STEPS, BATCH, 13))
        self.assert_shape(hs['hid2:out'].shape, (STEPS, BATCH, 14))
        self.assert_shape(hs['hid3:out'].shape, (STEPS, BATCH, 15))
        self.assert_shape(hs['out:out'].shape, (STEPS, BATCH, OUTS))


class TestPredictor(Base):
    def _build(self, *hiddens):
        return theanets.recurrent.Predictor((INS, ) + hiddens + (INS, ))

    def test_predict_onelayer(self):
        net = self._build(13)
        z = net.predict(self.probe)
        self.assert_shape(z.shape, (STEPS, BATCH, INS))

    def test_feed_forward(self):
        net = self._build(15, 13)
        hs = net.feed_forward(self.probe)
        assert len(hs) == 9, 'got {}'.format(list(hs.keys()))
        self.assert_shape(hs['in:out'].shape, (STEPS, BATCH, INS))
        self.assert_shape(hs['hid1:out'].shape, (STEPS, BATCH, 15))
        self.assert_shape(hs['hid2:out'].shape, (STEPS, BATCH, 13))
        self.assert_shape(hs['out:out'].shape, (STEPS, BATCH, INS))


class TestClassifier(Base):
    def _build(self, *hiddens):
        return theanets.recurrent.Classifier((INS, ) + hiddens + (OUTS, ))

    def test_predict_onelayer(self):
        net = self._build(13)
        z = net.predict(self.probe)
        self.assert_shape(z.shape, (STEPS, BATCH))

    def test_score_onelayer(self):
        net = self._build(13)
        z = net.score(self.probe, self.probe_classes)
        assert 0 < z < 1

    def test_predict_proba_onelayer(self):
        net = self._build(13)
        z = net.predict_proba(self.probe)
        self.assert_shape(z.shape, (STEPS, BATCH, OUTS))

    def test_predict_twolayer(self):
        net = self._build(13, 14)
        z = net.predict(self.probe)
        self.assert_shape(z.shape, (STEPS, BATCH))

    def test_feed_forward(self):
        net = self._build(15, 13)
        hs = net.feed_forward(self.probe)
        assert len(hs) == 9, 'got {}'.format(list(hs.keys()))
        self.assert_shape(hs['in:out'].shape, (STEPS, BATCH, INS))
        self.assert_shape(hs['hid1:out'].shape, (STEPS, BATCH, 15))
        self.assert_shape(hs['hid2:out'].shape, (STEPS, BATCH, 13))
        self.assert_shape(hs['out:out'].shape, (STEPS, BATCH, OUTS))


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

    def test_feed_forward(self):
        net = self._build(15, 13)
        hs = net.feed_forward(self.probe)
        assert len(hs) == 9, 'got {}'.format(list(hs.keys()))
        self.assert_shape(hs['in:out'].shape, (STEPS, BATCH, INS))
        self.assert_shape(hs['hid1:out'].shape, (STEPS, BATCH, 15))
        self.assert_shape(hs['hid2:out'].shape, (STEPS, BATCH, 13))
        self.assert_shape(hs['out:out'].shape, (STEPS, BATCH, INS))

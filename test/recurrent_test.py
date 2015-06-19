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


class TestText:
    TXT = 'hello world, how are you!'

    def setUp(self):
        self.txt = theanets.recurrent.Text(self.TXT, alpha='helo wrd,!', unknown='_')

    def test_min_count(self):
        txt = theanets.recurrent.Text(self.TXT, min_count=2, unknown='_')
        assert txt.text == 'hello worl__ how _re _o__'
        assert txt.alpha == ' ehlorw'

        txt = theanets.recurrent.Text(self.TXT, min_count=3, unknown='_')
        assert txt.text == '__llo _o_l__ _o_ ___ _o__'
        assert txt.alpha == ' lo'

    def test_alpha(self):
        assert self.txt.text == 'hello world, how _re _o_!'
        assert self.txt.alpha == 'helo wrd,!'

    def test_encode(self):
        assert self.txt.encode('hello!') == [1, 2, 3, 3, 4, 10]
        assert self.txt.encode('you!') == [0, 4, 0, 10]

    def test_decode(self):
        assert self.txt.decode([1, 2, 3, 3, 4, 10]) == 'hello!'
        assert self.txt.decode([0, 4, 0, 10]) == '_o_!'

    def test_classifier_batches(self):
        b = self.txt.classifier_batches(3, 2)
        assert len(b()) == 2
        assert b()[0].shape == (3, 2, 1 + len(self.txt.alpha))
        assert b()[1].shape == (3, 2)
        assert not np.allclose(b()[0], b()[0])


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


class TestWeightedRegressor(TestRegressor):
    def _build(self, *hiddens):
        return theanets.recurrent.Regressor(
            layers=(INS, ) + hiddens + (OUTS, ), weighted=True)


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


class TestWeightedClassifier(TestClassifier):
    def _build(self, *hiddens):
        return theanets.recurrent.Classifier(
            layers=(INS, ) + hiddens + (OUTS, ), weighted=True)


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


class TestWeightedAutoencoder(TestAutoencoder):
    def _build(self, *hiddens):
        return theanets.recurrent.Autoencoder(
            (INS, ) + hiddens + (INS, ), weighted=True)

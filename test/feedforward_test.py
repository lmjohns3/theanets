import numpy as np
import theanets

import util


class TestRegressor(util.Base):
    def _build(self, *hiddens):
        return theanets.Regressor(
            [self.NUM_INPUTS] + list(hiddens) + [self.NUM_OUTPUTS])

    def test_sgd(self):
        self.exp = theanets.Experiment(
            theanets.Regressor,
            layers=[self.NUM_INPUTS, 10, self.NUM_OUTPUTS])
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


class TestWeightedRegressor(TestRegressor):
    def _build(self, *hiddens):
        return theanets.Regressor(
            [self.NUM_INPUTS] + list(hiddens) + [self.NUM_OUTPUTS],
            weighted=True)

    def test_sgd(self):
        self.exp = theanets.Experiment(
            theanets.Regressor,
            layers=[self.NUM_INPUTS, 10, self.NUM_OUTPUTS],
            weighted=True)
        self.assert_progress('sgd', [self.INPUTS, self.OUTPUTS, self.OUTPUT_WEIGHTS])


class TestClassifier(util.Base):
    def _build(self, *hiddens):
        return theanets.Classifier(
            [self.NUM_INPUTS] + list(hiddens) + [self.NUM_CLASSES])

    def test_sgd(self):
        self.exp = theanets.Experiment(
            theanets.Classifier,
            layers=(self.NUM_INPUTS, 10, self.NUM_CLASSES))
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

    def test_predict_logit_onelayer(self):
        net = self._build(13)
        z = net.predict_logit(self.INPUTS)
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


class TestWeightedClassifier(TestClassifier):
    def _build(self, *hiddens):
        return theanets.Classifier(
            [self.NUM_INPUTS] + list(hiddens) + [self.NUM_CLASSES],
            weighted=True)

    def test_sgd(self):
        self.exp = theanets.Experiment(
            theanets.Classifier,
            layers=(self.NUM_INPUTS, 10, self.NUM_CLASSES),
            weighted=True)
        self.assert_progress('sgd', [self.INPUTS, self.CLASSES, self.CLASS_WEIGHTS])

    def test_score_onelayer(self):
        net = self._build(13)
        w = 0.5 * np.ones(self.CLASSES.shape, 'f')
        z = net.score(self.INPUTS, self.CLASSES, w)
        assert 0 <= z <= 1


class TestAutoencoder(util.Base):
    def _build(self, *hiddens):
        return theanets.Autoencoder(
            [self.NUM_INPUTS] + list(hiddens) + [self.NUM_INPUTS])

    def test_sgd(self):
        self.exp = theanets.Experiment(
            theanets.Autoencoder,
            layers=(self.NUM_INPUTS, 10, self.NUM_INPUTS))
        self.assert_progress('sgd', [self.INPUTS])

    def test_score_onelayer(self):
        net = self._build(13)
        z = net.score(self.INPUTS)
        assert z < 0

    def test_encode_onelayer(self):
        net = self._build(13)
        z = net.encode(self.INPUTS, 'hid1')
        self.assert_shape(z.shape, 13)

    def test_encode_twolayer(self):
        net = self._build(13, 14)
        z = net.encode(self.INPUTS)
        self.assert_shape(z.shape, 14)

    def test_encode_threelayer(self):
        net = self._build(13, 14, 15)
        z = net.encode(self.INPUTS)
        self.assert_shape(z.shape, 14)

    def test_decode_onelayer(self):
        net = self._build(13)
        x = net.decode(net.encode(self.INPUTS))
        self.assert_shape(x.shape, self.NUM_INPUTS)

    def test_decode_twolayer(self):
        net = self._build(13, 14)
        x = net.decode(net.encode(self.INPUTS))
        self.assert_shape(x.shape, self.NUM_INPUTS)

    def test_decode_threelayer(self):
        net = self._build(13, 14, 15)
        x = net.decode(net.encode(self.INPUTS))
        self.assert_shape(x.shape, self.NUM_INPUTS)

    def test_feed_forward(self):
        net = self._build(15, 13)
        hs = net.feed_forward(self.INPUTS)
        assert len(hs) == 9, 'got {}'.format(list(hs.keys()))
        self.assert_shape(hs['in:out'].shape, self.NUM_INPUTS)
        self.assert_shape(hs['hid1:out'].shape, 15)
        self.assert_shape(hs['out:out'].shape, self.NUM_INPUTS)


class TestWeightedAutoencoder(TestAutoencoder):
    def _build(self, *hiddens):
        return theanets.Autoencoder(
            [self.NUM_INPUTS] + list(hiddens) + [self.NUM_INPUTS],
            weighted=True)

    def test_sgd(self):
        self.exp = theanets.Experiment(
            theanets.Autoencoder,
            layers=(self.NUM_INPUTS, 10, self.NUM_INPUTS),
            weighted=True)
        self.assert_progress('sgd', [self.INPUTS, self.INPUT_WEIGHTS])

    def test_score_onelayer(self):
        net = self._build(13)
        labels = np.random.randint(0, 2, size=self.INPUTS.shape)
        z = net.score(self.INPUTS, labels)
        assert z < 0

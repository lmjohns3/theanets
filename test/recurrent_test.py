import numpy as np
import theanets

import util


class Base(util.Base):
    NUM_TIMES = 31
    NUM_EXAMPLES = util.Base.NUM_EXAMPLES
    NUM_INPUTS = util.Base.NUM_INPUTS
    NUM_OUTPUTS = util.Base.NUM_OUTPUTS
    NUM_CLASSES = util.Base.NUM_CLASSES

    INPUTS = np.random.randn(NUM_TIMES, NUM_EXAMPLES, NUM_INPUTS).astype('f')
    INPUT_WEIGHTS = np.random.randn(NUM_TIMES, NUM_EXAMPLES, NUM_INPUTS).astype('f')
    OUTPUTS = np.random.randn(NUM_TIMES, NUM_EXAMPLES, NUM_OUTPUTS).astype('f')
    OUTPUT_WEIGHTS = np.random.randn(NUM_TIMES, NUM_EXAMPLES, NUM_OUTPUTS).astype('f')
    CLASSES = np.random.randn(NUM_TIMES, NUM_EXAMPLES).astype('i')
    CLASS_WEIGHTS = np.random.rand(NUM_TIMES, NUM_EXAMPLES).astype('f')

    def assert_shape(self, actual, expected):
        if not isinstance(expected, tuple):
            expected = (self.NUM_TIMES, self.NUM_EXAMPLES, expected)
        assert actual == expected, 'expected {}, got {}'.format(expected, actual)


class TestFunctions(Base):
    def setUp(self):
        self.samples = np.random.randn(2 * self.NUM_TIMES, self.NUM_INPUTS)
        self.labels = np.random.randn(2 * self.NUM_TIMES, self.NUM_OUTPUTS)

    def test_batches_labeled(self):
        f = theanets.recurrent.batches(
            self.samples,
            self.labels,
            steps=self.NUM_TIMES,
            batch_size=self.NUM_EXAMPLES)
        assert len(f()) == 2
        assert f()[0].shape == (self.NUM_TIMES, self.NUM_EXAMPLES, self.NUM_INPUTS)
        assert f()[1].shape == (self.NUM_TIMES, self.NUM_EXAMPLES, self.NUM_OUTPUTS)

    def test_batches_unlabeled(self):
        f = theanets.recurrent.batches(
            self.samples, steps=self.NUM_TIMES, batch_size=self.NUM_EXAMPLES)
        assert len(f()) == 1
        assert f()[0].shape == (self.NUM_TIMES, self.NUM_EXAMPLES, self.NUM_INPUTS)


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


class TestRegressor(Base):
    def _build(self, *hiddens):
        return theanets.recurrent.Regressor(
            [self.NUM_INPUTS] + list(hiddens) + [self.NUM_OUTPUTS])

    def test_sgd(self):
        self.exp = theanets.Experiment(
            theanets.recurrent.Regressor,
            layers=(self.NUM_INPUTS, (10, 'rnn'), self.NUM_OUTPUTS))
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
        return theanets.recurrent.Regressor(
            [self.NUM_INPUTS] + list(hiddens) + [self.NUM_OUTPUTS],
            weighted=True)

    def test_sgd(self):
        self.exp = theanets.Experiment(
            theanets.recurrent.Regressor,
            layers=(self.NUM_INPUTS, (10, 'rnn'), self.NUM_OUTPUTS),
            weighted=True)
        self.assert_progress('sgd', [self.INPUTS, self.OUTPUTS, self.OUTPUT_WEIGHTS])


class TestClassifier(Base):
    def _build(self, *hiddens):
        return theanets.recurrent.Classifier(
            [self.NUM_INPUTS] + list(hiddens) + [self.NUM_OUTPUTS])

    def test_sgd(self):
        self.exp = theanets.Experiment(
            theanets.recurrent.Classifier,
            layers=(self.NUM_INPUTS, (10, 'rnn'), self.NUM_CLASSES))
        self.assert_progress('sgd', [self.INPUTS, self.CLASSES])

    def test_predict_onelayer(self):
        net = self._build(13)
        z = net.predict(self.INPUTS)
        self.assert_shape(z.shape, (self.NUM_TIMES, self.NUM_EXAMPLES))

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
        self.assert_shape(z.shape, (self.NUM_TIMES, self.NUM_EXAMPLES))

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
        return theanets.recurrent.Classifier(
            [self.NUM_INPUTS] + list(hiddens) + [self.NUM_OUTPUTS],
            weighted=True)

    def test_sgd(self):
        self.exp = theanets.Experiment(
            theanets.recurrent.Classifier,
            layers=(self.NUM_INPUTS, (10, 'rnn'), self.NUM_CLASSES),
            weighted=True)
        self.assert_progress('sgd', [self.INPUTS, self.CLASSES, self.CLASS_WEIGHTS])


class TestAutoencoder(Base):
    def _build(self, *hiddens):
        return theanets.recurrent.Autoencoder(
            [self.NUM_INPUTS] + list(hiddens) + [self.NUM_INPUTS])

    def test_sgd(self):
        self.exp = theanets.Experiment(
            theanets.recurrent.Autoencoder,
            layers=(self.NUM_INPUTS, (10, 'rnn'), self.NUM_INPUTS))
        self.assert_progress('sgd', [self.INPUTS])

    def test_encode_onelayer(self):
        net = self._build(13)
        z = net.predict(self.INPUTS)
        self.assert_shape(z.shape, self.NUM_INPUTS)

    def test_encode_twolayer(self):
        net = self._build(13, 14)
        z = net.predict(self.INPUTS)
        self.assert_shape(z.shape, self.NUM_INPUTS)

    def test_feed_forward(self):
        net = self._build(15, 13)
        hs = net.feed_forward(self.INPUTS)
        assert len(hs) == 9, 'got {}'.format(list(hs.keys()))
        self.assert_shape(hs['in:out'].shape, self.NUM_INPUTS)
        self.assert_shape(hs['hid1:out'].shape, 15)
        self.assert_shape(hs['hid2:out'].shape, 13)
        self.assert_shape(hs['out:out'].shape, self.NUM_INPUTS)


class TestWeightedAutoencoder(TestAutoencoder):
    def _build(self, *hiddens):
        return theanets.recurrent.Autoencoder(
            [self.NUM_INPUTS] + list(hiddens) + [self.NUM_INPUTS],
            weighted=True)

    def test_sgd(self):
        self.exp = theanets.Experiment(
            theanets.recurrent.Autoencoder,
            layers=(self.NUM_INPUTS, (10, 'rnn'), self.NUM_INPUTS),
            weighted=True)
        self.assert_progress('sgd', [self.INPUTS, self.INPUT_WEIGHTS])

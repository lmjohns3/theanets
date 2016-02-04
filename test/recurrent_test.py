import numpy as np
import pytest
import theanets

from util import RecurrentBase as Base


class TestFunctions(Base):
    @pytest.fixture
    def samples(self):
        return np.random.randn(2 * self.NUM_TIMES, self.NUM_INPUTS)

    @pytest.fixture
    def labels(self):
        return np.random.randn(2 * self.NUM_TIMES, self.NUM_OUTPUTS)

    def test_batches_labeled(self, samples, labels):
        f = theanets.recurrent.batches(
            [samples, labels], steps=self.NUM_TIMES, batch_size=self.NUM_EXAMPLES)
        assert len(f()) == 2
        assert f()[0].shape == (self.NUM_EXAMPLES, self.NUM_TIMES, self.NUM_INPUTS)
        assert f()[1].shape == (self.NUM_EXAMPLES, self.NUM_TIMES, self.NUM_OUTPUTS)

    def test_batches_unlabeled(self, samples):
        f = theanets.recurrent.batches(
            [samples], steps=self.NUM_TIMES, batch_size=self.NUM_EXAMPLES)
        assert len(f()) == 1
        assert f()[0].shape == (self.NUM_EXAMPLES, self.NUM_TIMES, self.NUM_INPUTS)


class TestText:
    TXT = 'hello world, how are you!'

    @pytest.fixture
    def txt(self):
        return theanets.recurrent.Text(self.TXT, alpha='helo wrd,!', unknown='_')

    def test_min_count(self):
        txt = theanets.recurrent.Text(self.TXT, min_count=2, unknown='_')
        assert txt.text == 'hello worl__ how _re _o__'
        assert txt.alpha == ' ehlorw'

        txt = theanets.recurrent.Text(self.TXT, min_count=3, unknown='_')
        assert txt.text == '__llo _o_l__ _o_ ___ _o__'
        assert txt.alpha == ' lo'

    def test_alpha(self, txt):
        assert txt.text == 'hello world, how _re _o_!'
        assert txt.alpha == 'helo wrd,!'

    def test_encode(self, txt):
        assert txt.encode('hello!') == [1, 2, 3, 3, 4, 10]
        assert txt.encode('you!') == [0, 4, 0, 10]

    def test_decode(self, txt):
        assert txt.decode([1, 2, 3, 3, 4, 10]) == 'hello!'
        assert txt.decode([0, 4, 0, 10]) == '_o_!'

    def test_classifier_batches(self, txt):
        b = txt.classifier_batches(steps=8, batch_size=5)
        assert len(b()) == 2
        assert b()[0].shape == (5, 8, 1 + len(txt.alpha))
        assert b()[1].shape == (5, 8)
        assert not np.allclose(b()[0], b()[0])


class TestRegressor(Base):
    def build(self, *hiddens):
        return theanets.recurrent.Regressor(
            [self.NUM_INPUTS] + list(hiddens) + [self.NUM_OUTPUTS])

    def test_sgd(self):
        self.assert_progress(
            self.build((10, 'rnn')), 'sgd', [self.INPUTS, self.OUTPUTS])

    def test_predict(self):
        net = self.build(15, 13)
        y = net.predict(self.INPUTS)
        self.assert_shape(y.shape, self.NUM_OUTPUTS)

    def test_feed_forward(self):
        net = self.build(15, 13)
        hs = net.feed_forward(self.INPUTS)
        assert len(hs) == 9, 'got {}'.format(list(hs.keys()))
        self.assert_shape(hs['in:out'].shape, self.NUM_INPUTS)
        self.assert_shape(hs['hid1:out'].shape, 15)
        self.assert_shape(hs['hid2:out'].shape, 13)
        self.assert_shape(hs['out:out'].shape, self.NUM_OUTPUTS)

    def test_multiple_recurrent(self):
        net = self.build(13, 14, 15)
        hs = net.feed_forward(self.INPUTS)
        assert len(hs) == 11, 'got {}'.format(list(hs.keys()))
        self.assert_shape(hs['in:out'].shape, self.NUM_INPUTS)
        self.assert_shape(hs['hid1:out'].shape, 13)
        self.assert_shape(hs['hid2:out'].shape, 14)
        self.assert_shape(hs['hid3:out'].shape, 15)
        self.assert_shape(hs['out:out'].shape, self.NUM_OUTPUTS)


class TestWeightedRegressor(TestRegressor):
    def build(self, *hiddens):
        return theanets.recurrent.Regressor(
            [self.NUM_INPUTS] + list(hiddens) + [self.NUM_OUTPUTS],
            weighted=True)

    def test_sgd(self):
        self.assert_progress(
            self.build((10, 'rnn')), 'sgd',
            [self.INPUTS, self.OUTPUTS, self.OUTPUT_WEIGHTS])


class TestClassifier(Base):
    def build(self, *hiddens):
        return theanets.recurrent.Classifier(
            [self.NUM_INPUTS] + list(hiddens) + [self.NUM_CLASSES])

    def test_sgd(self):
        self.assert_progress(
            self.build((10, 'rnn')), 'sgd', [self.INPUTS, self.CLASSES])

    def test_predict_onelayer(self):
        net = self.build(13)
        z = net.predict(self.INPUTS)
        self.assert_shape(z.shape, (self.NUM_EXAMPLES, self.NUM_TIMES))

    def test_score_onelayer(self):
        net = self.build(13)
        z = net.score(self.INPUTS, self.CLASSES)
        assert 0 <= z <= 1

    def test_predict_proba_onelayer(self):
        net = self.build(13)
        z = net.predict_proba(self.INPUTS)
        self.assert_shape(z.shape, self.NUM_CLASSES)

    def test_predict_twolayer(self):
        net = self.build(13, 14)
        z = net.predict(self.INPUTS)
        self.assert_shape(z.shape, (self.NUM_EXAMPLES, self.NUM_TIMES))

    def test_feed_forward(self):
        net = self.build(15, 13)
        hs = net.feed_forward(self.INPUTS)
        assert len(hs) == 9, 'got {}'.format(list(hs.keys()))
        self.assert_shape(hs['in:out'].shape, self.NUM_INPUTS)
        self.assert_shape(hs['hid1:out'].shape, 15)
        self.assert_shape(hs['hid2:out'].shape, 13)
        self.assert_shape(hs['out:out'].shape, self.NUM_CLASSES)

    def test_predict_sequence(self):
        net = self.build(13)

        count = 0
        for cs in net.predict_sequence([3, 0, 1, 2], 5, streams=3):
            assert isinstance(cs, list)
            assert len(cs) == 3
            count += 1
        assert count == 5

        count = 0
        for cs in net.predict_sequence([3, 0, 1, 2], 5):
            print(cs, type(cs))
            assert isinstance(cs, int)
            count += 1
        assert count == 5


class TestWeightedClassifier(TestClassifier):
    def build(self, *hiddens):
        return theanets.recurrent.Classifier(
            [self.NUM_INPUTS] + list(hiddens) + [self.NUM_CLASSES],
            weighted=True)

    def test_sgd(self):
        self.assert_progress(
            self.build((10, 'rnn')), 'sgd',
            [self.INPUTS, self.CLASSES, self.CLASS_WEIGHTS])


class TestAutoencoder(Base):
    def build(self, *hiddens):
        return theanets.recurrent.Autoencoder(
            [self.NUM_INPUTS] + list(hiddens) + [self.NUM_INPUTS])

    def test_sgd(self):
        self.assert_progress(self.build((10, 'rnn')), 'sgd', [self.INPUTS])

    def test_encode_onelayer(self):
        net = self.build(13)
        z = net.predict(self.INPUTS)
        self.assert_shape(z.shape, self.NUM_INPUTS)

    def test_encode_twolayer(self):
        net = self.build(13, 14)
        z = net.predict(self.INPUTS)
        self.assert_shape(z.shape, self.NUM_INPUTS)

    def test_feed_forward(self):
        net = self.build(15, 13)
        hs = net.feed_forward(self.INPUTS)
        assert len(hs) == 9, 'got {}'.format(list(hs.keys()))
        self.assert_shape(hs['in:out'].shape, self.NUM_INPUTS)
        self.assert_shape(hs['hid1:out'].shape, 15)
        self.assert_shape(hs['hid2:out'].shape, 13)
        self.assert_shape(hs['out:out'].shape, self.NUM_INPUTS)


class TestWeightedAutoencoder(TestAutoencoder):
    def build(self, *hiddens):
        return theanets.recurrent.Autoencoder(
            [self.NUM_INPUTS] + list(hiddens) + [self.NUM_INPUTS],
            weighted=True)

    def test_sgd(self):
        self.assert_progress(
            self.build((10, 'rnn')), 'sgd', [self.INPUTS, self.INPUT_WEIGHTS])

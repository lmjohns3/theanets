import numpy as np
import pytest
import theanets

import util as u

AE_LAYERS = [u.NUM_INPUTS, (u.NUM_HID1, 'rnn'), (u.NUM_HID2, 'rnn'), u.NUM_INPUTS]
CLF_LAYERS = [u.NUM_INPUTS, (u.NUM_HID1, 'rnn'), (u.NUM_HID2, 'rnn'), u.NUM_CLASSES]
REG_LAYERS = [u.NUM_INPUTS, (u.NUM_HID1, 'rnn'), (u.NUM_HID2, 'rnn'), u.NUM_OUTPUTS]


def assert_shape(actual, expected):
    if not isinstance(expected, tuple):
        expected = (u.NUM_EXAMPLES, u.RNN.NUM_TIMES, expected)
    assert actual == expected


@pytest.mark.parametrize('Model, layers, weighted, data', [
    (theanets.recurrent.Regressor, REG_LAYERS, False, u.RNN.REG_DATA),
    (theanets.recurrent.Classifier, CLF_LAYERS, False, u.RNN.CLF_DATA),
    (theanets.recurrent.Autoencoder, AE_LAYERS, False, u.RNN.AE_DATA),
    (theanets.recurrent.Regressor, REG_LAYERS, True, u.RNN.WREG_DATA),
    (theanets.recurrent.Classifier, CLF_LAYERS, True, u.RNN.WCLF_DATA),
    (theanets.recurrent.Autoencoder, AE_LAYERS, True, u.RNN.WAE_DATA),
])
def test_sgd(Model, layers, weighted, data):
    u.assert_progress(Model(layers, weighted=weighted), data)


@pytest.mark.parametrize('Model, layers', [
    (theanets.recurrent.Regressor, REG_LAYERS),
    (theanets.recurrent.Classifier, CLF_LAYERS),
    (theanets.recurrent.Autoencoder, AE_LAYERS),
])
def test_predict(Model, layers):
    assert_shape(Model(layers).predict(u.INPUTS).shape, output)


@pytest.mark.parametrize('Model, layers, target, score', [
    (theanets.recurrent.Regressor, REG_LAYERS, u.RNN.OUTPUTS, -0.73883247375488281),
    (theanets.recurrent.Classifier, CLF_LAYERS, u.RNN.CLASSES, 0.0020161290322580645),
    (theanets.recurrent.Autoencoder, AE_LAYERS, u.RNN.INPUTS, 81.411415100097656),
])
def test_score(Model, layers, target, score):
    assert Model(layers).score(u.RNN.INPUTS, target) == score


@pytest.mark.parametrize('Model, layers, target', [
    (theanets.recurrent.Regressor, REG_LAYERS, u.NUM_OUTPUTS),
    (theanets.recurrent.Classifier, CLF_LAYERS, u.NUM_CLASSES),
    (theanets.recurrent.Autoencoder, AE_LAYERS, u.NUM_INPUTS),
])
def test_predict(Model, layers, target):
    outs = Model(layers).feed_forward(u.RNN.INPUTS)
    assert len(list(outs)) == 7
    assert_shape(outs['in:out'].shape, u.NUM_INPUTS)
    assert_shape(outs['hid1:out'].shape, u.NUM_HID1)
    assert_shape(outs['hid2:out'].shape, u.NUM_HID2)
    assert_shape(outs['out:out'].shape, target)


def test_symbolic_initial_state():
    net = theanets.recurrent.Regressor([
        dict(size=u.NUM_INPUTS, form='input', name='h0', ndim=2),
        dict(size=u.NUM_INPUTS, form='input', name='in'),
        dict(size=u.NUM_HID1, form='rnn', name='rnn', h_0='h0'),
        dict(size=u.NUM_OUTPUTS, form='ff', name='out'),
    ])
    H0 = np.random.randn(u.NUM_EXAMPLES, u.NUM_HID1).astype('f')
    u.assert_progress(net, [H0, u.RNN.INPUTS, u.RNN.OUTPUTS])


class TestClassifier:
    @pytest.fixture
    def net(self):
        return theanets.recurrent.Classifier(CLF_LAYERS)

    def test_predict_proba(self, net):
        assert_shape(net.predict_proba(u.RNN.INPUTS).shape, u.NUM_CLASSES)

    def test_predict_logit(self, net):
        assert_shape(net.predict_logit(u.RNN.INPUTS).shape, u.NUM_CLASSES)

    def test_score(self, net):
        w = 0.5 * np.ones(u.CLASSES.shape, 'f')
        assert 0 <= net.score(u.RNN.INPUTS, u.CLASSES, w) <= 1

    def test_predict_sequence(self, net):
        assert list(net.predict_sequence([0, 1, 2], 5, rng=13)) == [4, 5, 1, 3, 1]


class TestAutoencoder:
    @pytest.fixture
    def net(self):
        return theanets.recurrent.Autoencoder(AE_LAYERS)

    def test_encode_hid1(self, net):
        z = net.encode(u.RNN.INPUTS, 'hid1')
        assert_shape(z.shape, u.NUM_HID1)

    def test_encode_hid2(self, net):
        z = net.encode(u.RNN.INPUTS, 'hid2')
        assert_shape(z.shape, u.NUM_HID2)

    def test_decode_hid1(self, net):
        x = net.decode(net.encode(u.RNN.INPUTS))
        assert_shape(x.shape, u.NUM_INPUTS)

    def test_decode_hid2(self, net):
        x = net.decode(net.encode(u.RNN.INPUTS, 'hid2'), 'hid2')
        assert_shape(x.shape, u.NUM_INPUTS)

    def test_score(self, net):
        labels = np.random.randint(0, 2, size=u.RNN.INPUTS.shape)
        assert net.score(u.RNN.INPUTS, labels) < 0


class TestFunctions:
    @pytest.fixture
    def samples(self):
        return np.random.randn(2 * u.RNN.NUM_TIMES, u.NUM_INPUTS)

    @pytest.fixture
    def labels(self):
        return np.random.randn(2 * u.RNN.NUM_TIMES, u.NUM_OUTPUTS)

    def test_batches_labeled(self, samples, labels):
        f = theanets.recurrent.batches(
            [samples, labels], steps=u.RNN.NUM_TIMES, batch_size=u.NUM_EXAMPLES)
        assert len(f()) == 2
        assert_shape(f()[0].shape, u.NUM_INPUTS)
        assert_shape(f()[1].shape, u.NUM_OUTPUTS)

    def test_batches_unlabeled(self, samples):
        f = theanets.recurrent.batches(
            [samples], steps=u.RNN.NUM_TIMES, batch_size=u.NUM_EXAMPLES)
        assert len(f()) == 1
        assert_shape(f()[0].shape, u.NUM_INPUTS)


class TestText:
    TXT = 'hello world, how are you!'

    def test_min_count(self):
        txt = theanets.recurrent.Text(self.TXT, min_count=2, unknown='_')
        assert txt.text == 'hello worl__ how _re _o__'
        assert txt.alpha == ' ehlorw'

        txt = theanets.recurrent.Text(self.TXT, min_count=3, unknown='_')
        assert txt.text == '__llo _o_l__ _o_ ___ _o__'
        assert txt.alpha == ' lo'

    @pytest.fixture
    def txt(self):
        return theanets.recurrent.Text(self.TXT, alpha='helo wrd,!', unknown='_')

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

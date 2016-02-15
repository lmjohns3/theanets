import numpy as np
import pytest
import theanets

import util as u


@pytest.mark.parametrize('Model, layers, weighted, data', [
    (theanets.Regressor, u.REG_LAYERS, False, u.REG_DATA),
    (theanets.Classifier, u.CLF_LAYERS, False, u.CLF_DATA),
    (theanets.Autoencoder, u.AE_LAYERS, False, u.AE_DATA),
    (theanets.Regressor, u.REG_LAYERS, True, u.WREG_DATA),
    (theanets.Classifier, u.CLF_LAYERS, True, u.WCLF_DATA),
    (theanets.Autoencoder, u.AE_LAYERS, True, u.WAE_DATA),
])
def test_sgd(Model, layers, weighted, data):
    u.assert_progress(Model(layers, weighted=weighted), data)


@pytest.mark.parametrize('Model, layers, output', [
    (theanets.Regressor, u.REG_LAYERS, u.NUM_OUTPUTS),
    (theanets.Classifier, u.CLF_LAYERS, (u.NUM_EXAMPLES, )),
    (theanets.Autoencoder, u.AE_LAYERS, u.NUM_INPUTS),
])
def test_predict(Model, layers, output):
    u.assert_shape(Model(layers).predict(u.INPUTS).shape, output)


@pytest.mark.parametrize('Model, layers, target, score', [
    (theanets.Regressor, u.REG_LAYERS, u.OUTPUTS, -1.0473043918609619),
    (theanets.Classifier, u.CLF_LAYERS, u.CLASSES, 0.171875),
    (theanets.Autoencoder, u.AE_LAYERS, u.INPUTS, 15.108331680297852),
])
def test_score(Model, layers, target, score):
    assert Model(layers).score(u.INPUTS, target) == score


@pytest.mark.parametrize('Model, layers, target', [
    (theanets.Regressor, u.REG_LAYERS, u.NUM_OUTPUTS),
    (theanets.Classifier, u.CLF_LAYERS, u.NUM_CLASSES),
    (theanets.Autoencoder, u.AE_LAYERS, u.NUM_INPUTS),
])
def test_feed_forward(Model, layers, target):
    outs = Model(layers).feed_forward(u.INPUTS)
    assert len(list(outs)) == 7
    u.assert_shape(outs['in:out'].shape, u.NUM_INPUTS)
    u.assert_shape(outs['hid1:out'].shape, u.NUM_HID1)
    u.assert_shape(outs['hid2:out'].shape, u.NUM_HID2)
    u.assert_shape(outs['out:out'].shape, target)


def test_decode_from_multiple_layers():
    net = theanets.Regressor([u.NUM_INPUTS, u.NUM_HID1, u.NUM_HID2, dict(
        size=u.NUM_OUTPUTS, inputs=('hid2:out', 'hid1:out'))])
    outs = net.feed_forward(u.INPUTS)
    assert len(list(outs)) == 7
    u.assert_shape(outs['in:out'].shape, u.NUM_INPUTS)
    u.assert_shape(outs['hid1:out'].shape, u.NUM_HID1)
    u.assert_shape(outs['hid2:out'].shape, u.NUM_HID2)
    u.assert_shape(outs['out:out'].shape, u.NUM_OUTPUTS)


class TestClassifier:
    @pytest.fixture
    def net(self):
        return theanets.Classifier(u.CLF_LAYERS)

    def test_predict_proba(self, net):
        u.assert_shape(net.predict_proba(u.INPUTS).shape, u.NUM_CLASSES)

    def test_predict_logit(self, net):
        u.assert_shape(net.predict_logit(u.INPUTS).shape, u.NUM_CLASSES)

    def test_score(self, net):
        w = 0.5 * np.ones(u.CLASSES.shape, 'f')
        assert 0 <= net.score(u.INPUTS, u.CLASSES, w) <= 1


class TestAutoencoder:
    @pytest.fixture
    def net(self):
        return theanets.Autoencoder(u.AE_LAYERS)

    def test_encode_hid1(self, net):
        z = net.encode(u.INPUTS, 'hid1')
        u.assert_shape(z.shape, u.NUM_HID1)

    def test_encode_hid2(self, net):
        z = net.encode(u.INPUTS, 'hid2')
        u.assert_shape(z.shape, u.NUM_HID2)

    def test_decode_hid1(self, net):
        x = net.decode(net.encode(u.INPUTS))
        u.assert_shape(x.shape, u.NUM_INPUTS)

    def test_decode_hid2(self, net):
        x = net.decode(net.encode(u.INPUTS, 'hid2'), 'hid2')
        u.assert_shape(x.shape, u.NUM_INPUTS)

    def test_score(self, net):
        labels = np.random.randint(0, 2, size=u.INPUTS.shape)
        assert net.score(u.INPUTS, labels) < 0

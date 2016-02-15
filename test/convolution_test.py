import numpy as np
import pytest
import theanets

import util as u

REG_LAYERS = [
    u.NUM_INPUTS,
    dict(size=u.NUM_HID1, form='conv2', filter_size=u.CNN.FILTER_SIZE),
    dict(size=u.NUM_HID2, form='conv2', filter_size=u.CNN.FILTER_SIZE),
    ('flat', u.NUM_HID2 *
     (u.CNN.NUM_WIDTH - 2 * u.CNN.FILTER_WIDTH + 2) *
     (u.CNN.NUM_HEIGHT - 2 * u.CNN.FILTER_HEIGHT + 2)),
    u.NUM_OUTPUTS]

CLF_LAYERS = [
    u.NUM_INPUTS,
    dict(size=u.NUM_HID1, form='conv2', filter_size=u.CNN.FILTER_SIZE),
    dict(size=u.NUM_HID2, form='conv2', filter_size=u.CNN.FILTER_SIZE),
    ('flat', u.NUM_HID2 *
     (u.CNN.NUM_WIDTH - 2 * u.CNN.FILTER_WIDTH + 2) *
     (u.CNN.NUM_HEIGHT - 2 * u.CNN.FILTER_HEIGHT + 2)),
    u.NUM_CLASSES]


def assert_shape(actual, width, height, channels):
    assert actual == (u.NUM_EXAMPLES, width, height, channels)


@pytest.mark.parametrize('Model, layers, weighted, data', [
    (theanets.convolution.Regressor, REG_LAYERS, False, u.CNN.REG_DATA),
    (theanets.convolution.Classifier, CLF_LAYERS, False, u.CNN.CLF_DATA),
    (theanets.convolution.Regressor, REG_LAYERS, True, u.CNN.WREG_DATA),
    (theanets.convolution.Classifier, CLF_LAYERS, True, u.CNN.WCLF_DATA),
])
def test_sgd(Model, layers, weighted, data):
    u.assert_progress(Model(layers, weighted=weighted), data)


@pytest.mark.parametrize('Model, layers, output', [
    (theanets.convolution.Regressor, REG_LAYERS, u.NUM_OUTPUTS),
    (theanets.convolution.Classifier, CLF_LAYERS, (u.NUM_EXAMPLES, )),
])
def test_predict(Model, layers, output):
    u.assert_shape(Model(layers).predict(u.CNN.INPUTS).shape, output)


@pytest.mark.parametrize('Model, layers, target, score', [
    (theanets.convolution.Regressor, REG_LAYERS, u.OUTPUTS, -16.850263595581055),
    (theanets.convolution.Classifier, CLF_LAYERS, u.CLASSES, 0.171875),
])
def test_score(Model, layers, target, score):
    assert Model(layers).score(u.CNN.INPUTS, target) == score


@pytest.mark.parametrize('Model, layers, target', [
    (theanets.convolution.Regressor, REG_LAYERS, u.NUM_OUTPUTS),
    (theanets.convolution.Classifier, CLF_LAYERS, u.NUM_CLASSES),
])
def test_predict(Model, layers, target):
    outs = Model(layers).feed_forward(u.CNN.INPUTS)
    assert len(list(outs)) == 8
    W, H = u.CNN.NUM_WIDTH, u.CNN.NUM_HEIGHT
    w, h = u.CNN.FILTER_WIDTH, u.CNN.FILTER_HEIGHT
    assert_shape(outs['in:out'].shape, W, H, u.NUM_INPUTS)
    assert_shape(outs['hid1:out'].shape, W - w + 1, H - h + 1, u.NUM_HID1)
    assert_shape(outs['hid2:out'].shape, W - 2 * w + 2, H - 2 * h + 2, u.NUM_HID2)
    u.assert_shape(outs['out:out'].shape, target)


class TestClassifier:
    @pytest.fixture
    def net(self):
        return theanets.convolution.Classifier(CLF_LAYERS)

    def test_predict_proba(self, net):
        u.assert_shape(net.predict_proba(u.CNN.INPUTS).shape, u.NUM_CLASSES)

    def test_predict_logit(self, net):
        u.assert_shape(net.predict_logit(u.CNN.INPUTS).shape, u.NUM_CLASSES)

    def test_score(self, net):
        w = 0.5 * np.ones(u.CLASSES.shape, 'f')
        assert 0 <= net.score(u.CNN.INPUTS, u.CLASSES, w) <= 1

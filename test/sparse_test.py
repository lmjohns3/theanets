import numpy as np
import pytest
import scipy.sparse
import theanets

import util as u

CSR = dict(form='input', size=u.NUM_INPUTS, sparse='csr', name='in')
CSC = dict(form='input', size=u.NUM_INPUTS, sparse='csc', name='in')
REG_LAYERS = dict(csr=[CSR] + u.REG_LAYERS[1:], csc=[CSC] + u.REG_LAYERS[1:])
CLF_LAYERS = dict(csr=[CSR] + u.CLF_LAYERS[1:], csc=[CSC] + u.CLF_LAYERS[1:])
AE_LAYERS = dict(csr=[CSR] + u.AE_LAYERS[1:], csc=[CSC] + u.AE_LAYERS[1:])

CSR = scipy.sparse.csr_matrix(u.INPUTS)
CSC = scipy.sparse.csc_matrix(u.INPUTS)
INPUTS = dict(csr=CSR, csc=CSC)
REG_DATA = dict(csr=[CSR] + u.REG_DATA[1:], csc=[CSC] + u.REG_DATA[1:])
WREG_DATA = dict(csr=[CSR] + u.WREG_DATA[1:], csc=[CSC] + u.WREG_DATA[1:])
CLF_DATA = dict(csr=[CSR] + u.CLF_DATA[1:], csc=[CSC] + u.CLF_DATA[1:])
WCLF_DATA = dict(csr=[CSR] + u.WCLF_DATA[1:], csc=[CSC] + u.WCLF_DATA[1:])


@pytest.mark.parametrize('Model, layers, sparse, weighted, data', [
    (theanets.Regressor, REG_LAYERS, 'csr', True, WREG_DATA),
    (theanets.Classifier, CLF_LAYERS, 'csr', True, WCLF_DATA),
    (theanets.Regressor, REG_LAYERS, 'csc', True, WREG_DATA),
    (theanets.Classifier, CLF_LAYERS, 'csc', True, WCLF_DATA),
    (theanets.Regressor, REG_LAYERS, 'csr', False, REG_DATA),
    (theanets.Classifier, CLF_LAYERS, 'csr', False, CLF_DATA),
    (theanets.Regressor, REG_LAYERS, 'csc', False, REG_DATA),
    (theanets.Classifier, CLF_LAYERS, 'csc', False, CLF_DATA),
])
def test_sgd(Model, layers, sparse, weighted, data):
    u.assert_progress(Model(layers[sparse], weighted=weighted), data[sparse])


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


@pytest.mark.parametrize('Model, layers, sparse, target', [
    (theanets.Regressor, REG_LAYERS, 'csr', u.NUM_OUTPUTS),
    (theanets.Classifier, CLF_LAYERS, 'csr', u.NUM_CLASSES),
    (theanets.Autoencoder, AE_LAYERS, 'csr', u.NUM_INPUTS),
    (theanets.Regressor, REG_LAYERS, 'csc', u.NUM_OUTPUTS),
    (theanets.Classifier, CLF_LAYERS, 'csc', u.NUM_CLASSES),
    (theanets.Autoencoder, AE_LAYERS, 'csc', u.NUM_INPUTS),
])
def test_feed_forward(Model, layers, sparse, target):
    outs = Model(layers[sparse]).feed_forward(INPUTS[sparse])
    assert len(list(outs)) == 7
    u.assert_shape(outs['in:out'].shape, u.NUM_INPUTS)
    u.assert_shape(outs['hid1:out'].shape, u.NUM_HID1)
    u.assert_shape(outs['hid2:out'].shape, u.NUM_HID2)
    u.assert_shape(outs['out:out'].shape, target)

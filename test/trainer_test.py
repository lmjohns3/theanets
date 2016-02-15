import pytest
import theanets

import util as u


@pytest.fixture
def ae():
    return theanets.Autoencoder([
        u.NUM_INPUTS, u.NUM_HID1, u.NUM_HID2, u.NUM_INPUTS])


def test_downhill(ae):
    # this really tests that interaction with downhill works.
    u.assert_progress(ae, u.AE_DATA)


def test_layerwise(ae):
    u.assert_progress(ae, u.AE_DATA, algo='layerwise')


def test_layerwise_tied():
    ae = theanets.Autoencoder([
        u.NUM_INPUTS, u.NUM_HID1, u.NUM_HID2,
        (u.NUM_HID1, 'tied'), (u.NUM_INPUTS, 'tied')])
    u.assert_progress(ae, u.AE_DATA, algo='layerwise')


def test_sample(ae):
    trainer = ae.itertrain(u.AE_DATA, algo='sample')
    train0, valid0 = next(trainer)
    # for this trainer, we don't measure the loss.
    assert train0['loss'] == 0 == valid0['loss']


def test_unsupervised_pretrainer():
    u.assert_progress(
        theanets.Experiment(theanets.Classifier, u.CLF_LAYERS),
        u.AE_DATA, algo='pretrain')

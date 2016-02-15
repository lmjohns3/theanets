import pytest
import theanets

import util as u


class TestBuild:
    def test_mse(self):
        l = theanets.Loss.build('mse', target=2)
        assert callable(l)
        assert len(l.variables) == 1

    def test_mse_weighted(self):
        l = theanets.Loss.build('mse', target=2, weighted=True)
        assert callable(l)
        assert len(l.variables) == 2


@pytest.mark.parametrize('loss', ['xe', 'hinge'])
def test_classification(loss):
    net = theanets.Classifier([
        u.NUM_INPUTS, u.NUM_HID1, u.NUM_CLASSES], loss=loss)
    u.assert_progress(net, u.CLF_DATA)


@pytest.mark.parametrize('loss', ['mse', 'mae', 'mmd'])
def test_regression(loss):
    net = theanets.Regressor([
        u.NUM_INPUTS, u.NUM_HID1, u.NUM_OUTPUTS], loss=loss)
    u.assert_progress(net, u.REG_DATA)


def test_kl():
    net = theanets.Regressor([
        u.NUM_INPUTS, u.NUM_HID1, (u.NUM_OUTPUTS, 'softmax')], loss='kl')
    u.assert_progress(net, [u.INPUTS, abs(u.OUTPUTS)])


def test_gll():
    net = theanets.Regressor([
        u.NUM_INPUTS,
        dict(name='hid', size=u.NUM_HID1),
        dict(name='covar', activation='relu', inputs='hid', size=u.NUM_OUTPUTS),
        dict(name='mean', activation='linear', inputs='hid', size=u.NUM_OUTPUTS),
    ])
    net.set_loss('gll', target=2, mean_name='mean', covar_name='covar')
    u.assert_progress(net, u.REG_DATA)

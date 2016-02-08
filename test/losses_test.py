import pytest
import theanets

import util


class TestBuild:
    def test_build_mse(self):
        l = theanets.Loss.build('mse', target=2)
        assert callable(l)
        assert len(l.variables) == 1

    def test_build_mse_weighted(self):
        l = theanets.Loss.build('mse', target=2, weighted=True)
        assert callable(l)
        assert len(l.variables) == 2


class TestLosses(util.Base):
    @pytest.mark.parametrize('loss', ['xe', 'hinge'])
    def test_classification(self, loss):
        net = theanets.Classifier([
            self.NUM_INPUTS, 10, self.NUM_CLASSES], loss=loss)
        self.assert_progress(net, 'sgd', [self.INPUTS, self.CLASSES])

    @pytest.mark.parametrize('loss', ['mse', 'mae', 'mmd'])
    def test_regression(self, loss):
        net = theanets.Regressor([
            self.NUM_INPUTS, 10, self.NUM_OUTPUTS], loss=loss)
        self.assert_progress(net, 'sgd', [self.INPUTS, self.OUTPUTS])

    def test_kl(self):
        net = theanets.Regressor([
            self.NUM_INPUTS, 10, (self.NUM_OUTPUTS, 'softmax')], loss='kl')
        self.assert_progress(net, 'sgd', [self.INPUTS, abs(self.OUTPUTS)])

    def test_gll(self):
        kw = dict(inputs={'hid:out': 10}, size=self.NUM_OUTPUTS)
        net = theanets.Regressor([
            self.NUM_INPUTS,
            dict(name='hid', size=10),
            dict(name='covar', activation='relu', **kw),
            dict(name='mean', activation='linear', **kw),
        ])
        net.set_loss('gll', target=2, mean_name='mean', covar_name='covar')
        self.assert_progress(net, 'sgd', [self.INPUTS, self.OUTPUTS],
                             max_gradient_norm=1)

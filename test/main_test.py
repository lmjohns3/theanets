import numpy as np
import os
import tempfile

import theanets

import util


class TestExperiment(util.MNIST):
    def test_create_autoencoder(self):
        exp = theanets.Experiment(
            theanets.Autoencoder, layers=(self.DIGIT_SIZE, 2, self.DIGIT_SIZE))
        assert isinstance(exp.network, theanets.Autoencoder)

    def test_create_classifier(self):
        exp = theanets.Experiment(
            theanets.Classifier, layers=(self.DIGIT_SIZE, 2, 3))
        assert isinstance(exp.network, theanets.Classifier)

    def test_create_autoencoder(self):
        exp = theanets.Experiment(
            theanets.Regressor, layers=(self.DIGIT_SIZE, 2, 4))
        assert isinstance(exp.network, theanets.Regressor)

    def test_save_load(self):
        exp = theanets.Experiment(theanets.Autoencoder, layers=(10, 3, 4, 10))
        net = exp.network
        f, p = tempfile.mkstemp(suffix='pkl')
        os.close(f)
        os.unlink(p)
        try:
            exp.save(p)
            assert os.path.isfile(p)
            exp.load(p)
            assert exp.network is not net
            for lo, ln in zip(net.layers, exp.network.layers):
                assert lo.name == ln.name
                assert lo.inputs == ln.inputs
                assert lo.outputs == ln.outputs
        finally:
            if os.path.exists(p):
                os.unlink(p)

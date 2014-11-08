import numpy as np
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
        exp = theanets.Experiment(
            theanets.Autoencoder, layers=(10, 3, 4, 10))
        net = exp.network
        _, p = tempfile.mkstemp(suffix='pkl')
        try:
            assert not os.path.isfile(p)
            exp.save(p)
            assert os.path.isfile(p)
            exp.load(p)
            assert exp.network is not net
            assert exp.network.layers == (10, 3, 4, 10)
        finally:
            if os.path.exists(p):
                os.unlink(p)

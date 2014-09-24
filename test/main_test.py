import theanets
import numpy as np

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

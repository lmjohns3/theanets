import theanets

import util


class TestTrainer(util.MNIST):
    def assert_progress(self, algo, labels=None, **kwargs):
        data = [self.images]
        if labels is not None:
            data.append(labels)
        trainer = self.exp.itertrain(
            data, algorithm=algo, monitor_gradients=True, **kwargs)
        train0, valid0 = next(trainer)
        train1, valid1 = next(trainer)
        assert train1['loss'] < valid0['loss']   # should have made progress!
        assert valid1['loss'] == valid0['loss']  # no new validation occurred

    def test_sgd_autoencoder(self):
        # this really tests that interaction with downhill works.
        self.exp = theanets.Experiment(
            theanets.Autoencoder,
            layers=(self.DIGIT_SIZE, 10, self.DIGIT_SIZE))
        self.assert_progress('sgd')

    def test_sgd_classifier(self):
        self.exp = theanets.Experiment(
            theanets.Classifier,
            layers=(self.DIGIT_SIZE, 10, 10))
        self.assert_progress('sgd', labels=self.labels)

    def test_sgd_regressor(self):
        self.exp = theanets.Experiment(
            theanets.Regressor,
            layers=(self.DIGIT_SIZE, 10, 1))
        self.assert_progress('sgd', labels=self.labels[:, None].astype('f'))

    def test_layerwise(self):
        self.exp = theanets.Experiment(
            theanets.Autoencoder,
            layers=(self.DIGIT_SIZE, 10, 10, self.DIGIT_SIZE))
        self.assert_progress('layerwise')

    def test_sample(self):
        exp = theanets.Experiment(
            theanets.Autoencoder,
            layers=(self.DIGIT_SIZE, 10, 10, self.DIGIT_SIZE))
        trainer = exp.itertrain(
            self.images, algorithm='sample', monitor_gradients=True)
        train0, valid0 = next(trainer)
        # for this trainer, we don't measure the loss.
        assert train0['loss'] == 0 == valid0['loss']

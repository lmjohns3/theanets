import theanets

import util


class TestTrainer(util.MNIST):
    def setUp(self):
        super(TestTrainer, self).setUp()
        self.exp = theanets.Experiment(
            theanets.Autoencoder,
            layers=(self.DIGIT_SIZE, 10, self.DIGIT_SIZE))

    def assert_progress(self, algo, **kwargs):
        trainer = self.exp.itertrain(self.images, optimize=algo, **kwargs)
        train, valid = next(trainer)
        assert train['loss'] < valid['loss']

    def test_sgd(self):
        self.assert_progress('sgd', learning_rate=1e-4)

    def test_nag(self):
        self.assert_progress('nag', learning_rate=1e-4)

    def test_rprop(self):
        self.assert_progress('rprop', learning_rate=1e-4)

    def test_rmsprop(self):
        self.assert_progress('rmsprop', learning_rate=1e-4)

    def test_adadelta(self):
        self.assert_progress('adadelta', learning_rate=1e-4)

    def test_cg(self):
        self.assert_progress('cg')

    def test_layerwise(self):
        self.exp = theanets.Experiment(
            theanets.Autoencoder,
            layers=(self.DIGIT_SIZE, 10, 10, self.DIGIT_SIZE))
        self.assert_progress('layerwise')

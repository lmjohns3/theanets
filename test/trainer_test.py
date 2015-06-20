import theanets

import util


class TestTrainer(util.Base):
    def setUp(self):
        self.exp = theanets.Experiment(
            theanets.Autoencoder,
            [self.NUM_INPUTS, 10, 10, self.NUM_INPUTS])

    def test_downhill(self):
        # this really tests that interaction with downhill works.
        self.assert_progress('sgd', [self.INPUTS])

    def test_layerwise(self):
        self.assert_progress('layerwise', [self.INPUTS])

    def test_sample(self):
        trainer = self.exp.itertrain(
            self.INPUTS, algorithm='sample', monitor_gradients=True)
        train0, valid0 = next(trainer)
        # for this trainer, we don't measure the loss.
        assert train0['loss'] == 0 == valid0['loss']

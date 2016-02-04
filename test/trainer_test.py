import theanets

import util


class TestTrainer(util.Base):
    def build(self):
        return theanets.Autoencoder([self.NUM_INPUTS, 10, self.NUM_INPUTS])

    def test_downhill(self):
        # this really tests that interaction with downhill works.
        self.assert_progress(self.build(), 'sgd', [self.INPUTS])

    def test_layerwise(self):
        self.assert_progress(self.build(), 'layerwise', [self.INPUTS])

    def test_sample(self):
        trainer = self.build().itertrain(
            [self.INPUTS], algorithm='sample', monitor_gradients=True)
        train0, valid0 = next(trainer)
        # for this trainer, we don't measure the loss.
        assert train0['loss'] == 0 == valid0['loss']

    def test_unsupervised_pretrainer(self):
        exp = theanets.Experiment(
            theanets.Classifier,
            [self.NUM_INPUTS, 10, 20, self.NUM_CLASSES])
        self.assert_progress(exp, 'pretrain', [self.INPUTS])

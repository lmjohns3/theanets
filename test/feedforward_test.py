import theanets
import numpy as np
import skdata.mnist


class MNIST:
    NUM_DIGITS = 100

    def setUp(self):
        mnist = skdata.mnist.dataset.MNIST()
        mnist.meta  # trigger download if needed.
        def arr(n, dtype):
            arr = mnist.arrays[n]
            return arr.reshape((len(arr), -1)).astype(dtype)
        self.images = arr('train_images', 'f')[:MNIST.NUM_DIGITS] / 255.
        self.labels = arr('train_labels', 'b')[:MNIST.NUM_DIGITS]


class TestNetwork(MNIST):
    def _build(self, *hiddens, **kwargs):
        return theanets.Regressor(
            layers=(784, ) + hiddens,
            activation='logistic',
            **kwargs)

    def test_predict(self):
        net = self._build(15, 13)
        y = net.predict(self.images)
        assert y.shape == (self.NUM_DIGITS, 13)

    def test_feed_forward(self):
        net = self._build(15, 13)
        hs = net.feed_forward(self.images)
        assert len(hs) == 2
        assert hs[0].shape == (self.NUM_DIGITS, 15)
        assert hs[1].shape == (self.NUM_DIGITS, 13)


class TestClassifier(MNIST):
    def _build(self, *hiddens, **kwargs):
        return theanets.Classifier(
            layers=(784, ) + hiddens + (10, ),
            activation='logistic',
            **kwargs)

    def test_classify_onelayer(self):
        net = self._build(13)
        z = net.classify(self.images)
        assert z.shape == (self.NUM_DIGITS, )

    def test_classify_twolayer(self):
        net = self._build(13, 14)
        z = net.classify(self.images)
        assert z.shape == (self.NUM_DIGITS, )


class TestAutoencoder(MNIST):
    def _build(self, *hiddens, **kwargs):
        return theanets.Autoencoder(
            layers=(784, ) + hiddens + (784, ),
            activation='logistic',
            **kwargs)

    def test_encode_onelayer(self):
        net = self._build(13)
        z = net.encode(self.images)
        assert z.shape == (self.NUM_DIGITS, 13)

    def test_encode_twolayer(self):
        net = self._build(13, 14)
        z = net.encode(self.images)
        assert z.shape == (self.NUM_DIGITS, 14)

    def test_encode_threelayer(self):
        net = self._build(13, 14, 15)
        z = net.encode(self.images)
        assert z.shape == (self.NUM_DIGITS, 14)

    def test_decode_onelayer(self):
        net = self._build(13)
        x = net.decode(net.encode(self.images))
        assert x.shape == (self.NUM_DIGITS, 784)

    def test_decode_twolayer(self):
        net = self._build(13, 14)
        x = net.decode(net.encode(self.images))
        assert x.shape == (self.NUM_DIGITS, 784)

    def test_decode_threelayer(self):
        net = self._build(13, 14, 15)
        x = net.decode(net.encode(self.images))
        assert x.shape == (self.NUM_DIGITS, 784)
        #err = ((valid - xhat) ** 2).sum(axis=1).mean()

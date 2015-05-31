import theanets

import util


class TestClassifier(util.MNIST):
    def _build(self, *hiddens):
        return theanets.Classifier((self.DIGIT_SIZE, ) + hiddens + (10, ))

    def test_predict_onelayer(self):
        net = self._build(13)
        z = net.predict(self.images)
        assert z.shape == (self.NUM_DIGITS, )

    def test_score_onelayer(self):
        net = self._build(13)
        z = net.score(self.images, self.labels)
        assert 0 < z < 1

    def test_predict_proba_onelayer(self):
        net = self._build(13)
        z = net.predict_proba(self.images)
        assert z.shape == (self.NUM_DIGITS, 10)

    def test_predict_twolayer(self):
        net = self._build(13, 14)
        z = net.predict(self.images)
        assert z.shape == (self.NUM_DIGITS, )


class TestAutoencoder(util.MNIST):
    def _build(self, *hiddens):
        return theanets.Autoencoder(
            (self.DIGIT_SIZE, ) + hiddens + (self.DIGIT_SIZE, ))

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
        assert x.shape == (self.NUM_DIGITS, self.DIGIT_SIZE)

    def test_decode_twolayer(self):
        net = self._build(13, 14)
        x = net.decode(net.encode(self.images))
        assert x.shape == (self.NUM_DIGITS, self.DIGIT_SIZE)

    def test_decode_threelayer(self):
        net = self._build(13, 14, 15)
        x = net.decode(net.encode(self.images))
        assert x.shape == (self.NUM_DIGITS, self.DIGIT_SIZE)

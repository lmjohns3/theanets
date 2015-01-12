import numpy as np
import theanets
import theano.tensor as TT


class TestLayer:
    def test_build(self):
        for f in 'feedforward Feedforward classifier rnn lstm'.split():
            l = theanets.layers.build(f, nin=2, nout=4)
            assert isinstance(l, theanets.layers.Layer)


class Base:
    def setUp(self):
        self.x = TT.matrix('x')


class TestFeedforward(Base):
    def _build(self):
        return theanets.layers.Feedforward(nin=2, nout=4)

    def test_create(self):
        l = self._build()
        assert l.reset() == 12

    def test_transform(self):
        l = self._build()
        l.reset()
        out, upd = l.transform(self.x)
        assert out is not None
        assert not upd


class TestTied(Base):
    def _build(self):
        l0 = theanets.layers.Feedforward(nin=2, nout=4)
        l0.reset()
        return theanets.layers.Tied(partner=l0)

    def test_create(self):
        l = self._build()
        assert l.reset() == 2

    def test_transform(self):
        l = self._build()
        l.reset()
        out, upd = l.transform(self.x)
        assert out is not None
        assert not upd


class TestClassifier(Base):
    def _build(self):
        return theanets.layers.Classifier(nin=2, nout=4)

    def test_create(self):
        l = self._build()
        assert l.reset() == 12

    def test_transform(self):
        l = self._build()
        l.reset()
        out, upd = l.transform(self.x)
        assert out is not None
        assert not upd


class TestRNN(Base):
    def _build(self, **kwargs):
        return theanets.layers.RNN(nin=2, nout=4, **kwargs)

    def test_create(self):
        l = self._build(viscosity='lin')
        assert l.reset() == 28

    def test_transform(self):
        l = self._build()
        l.reset()
        out, upd = l.transform(self.x)
        assert out is not None
        assert not upd


class TestMRNN(Base):
    def _build(self):
        return theanets.layers.MRNN(nin=2, nout=4, factors=3)

    def test_create(self):
        l = self._build()
        assert l.reset() == 42

    def test_transform(self):
        l = self._build()
        l.reset()
        out, upd = l.transform(self.x)
        assert out is not None
        assert not upd


class TestLSTM(Base):
    def _build(self):
        return theanets.layers.LSTM(nin=2, nout=4)

    def test_create(self):
        l = self._build()
        assert l.reset() == 124

    def test_transform(self):
        l = self._build()
        l.reset()
        out, upd = l.transform(self.x)
        assert out is not None
        assert not upd

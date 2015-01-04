import theanets
import numpy as np


class TestLayer:
    def test_build(self):
        layer = theanets.layers.build('feedforward', nin=2, nout=4)
        assert isinstance(layer, theanets.layers.Layer)


class TestFeedforward:
    def test_create(self):
        l = theanets.layers.Feedforward(nin=2, nout=4)
        assert l.reset() == 12

class TestTied:
    def test_create(self):
        l0 = theanets.layers.Feedforward(nin=2, nout=4)
        l = theanets.layers.Tied(partner=l0)
        assert l.reset() == 2

class TestClassifier:
    def test_create(self):
        l = theanets.layers.Classifier(nin=2, nout=4)
        assert l.reset() == 12

class TestRecurrent:
    def test_create(self):
        l = theanets.layers.Recurrent(nin=2, nout=4)
        assert l.reset() == 28

class TestMRNN:
    def test_create(self):
        l = theanets.layers.MRNN(nin=2, nout=4, factors=3)
        assert l.reset() == 42

class TestLSTM:
    def test_create(self):
        l = theanets.layers.LSTM(nin=2, nout=4)
        assert l.reset() == 124

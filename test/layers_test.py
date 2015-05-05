import numpy as np
import theanets
import theano.tensor as TT


class Base:
    def setUp(self):
        self.x = TT.matrix('x')
        self.l = self._build()

    def assert_param_names(self, expected):
        assert (sorted(p.name for p in self.l.params) ==
                sorted('l_{}'.format(n) for n in expected))

    def assert_count(self, expected):
        real = self.l.num_params
        assert real == expected, 'got {}, expected {}'.format(real, expected)

    def assert_spec(self, **expected):
        real = self.l.to_spec()
        err = 'got {}, expected {}'.format(real, expected)
        for k, v in expected.items():
            try:
                r = real[k]
            except KeyError:
                assert False, err
            if isinstance(v, np.ndarray) or isinstance(r, np.ndarray):
                assert np.allclose(r, v), err
            else:
                assert r == v, err


class TestLayer(Base):
    def _build(self):
        return theanets.layers.Feedforward(inputs=2, size=4, name='l')

    def test_build(self):
        for f in 'feedforward Feedforward classifier rnn lstm'.split():
            l = theanets.layers.build(f, inputs=2, size=4)
            assert isinstance(l, theanets.layers.Layer)

    def test_connect(self):
        out, mon, upd = self.l.connect(dict(out=self.x), monitors=(0.1, 0.2))
        assert len(out) == 2
        assert len(mon) == 2
        assert len(upd) == 0


class TestFeedforward(Base):
    def _build(self):
        return theanets.layers.Feedforward(inputs=2, size=4, name='l')

    def test_create(self):
        self.assert_param_names(['w', 'b'])
        self.assert_count(12)

    def test_transform(self):
        out, upd = self.l.transform(dict(out=self.x))
        assert len(out) == 2
        assert not upd


class TestMultiFeedforward(Base):
    def _build(self):
        return theanets.layers.Feedforward(inputs=dict(a=2, b=2), size=4, name='l')

    def test_create(self):
        self.assert_param_names(['w_a', 'w_b', 'b'])
        self.assert_count(20)

    def test_transform(self):
        out, upd = self.l.transform(dict(a=self.x, b=self.x))
        assert len(out) == 2
        assert not upd


class TestTied(Base):
    def _build(self):
        l0 = theanets.layers.Feedforward(inputs=2, size=4, name='l0')
        return theanets.layers.Tied(partner=l0, name='l')

    def test_create(self):
        l = self._build()
        assert sorted(p.name for p in l.params) == [l.name + '_b']
        self.assert_count(2)

    def test_transform(self):
        out, upd = self.l.transform(dict(out=self.x))
        assert len(out) == 2
        assert not upd


class TestClassifier(Base):
    def _build(self):
        return theanets.layers.Classifier(inputs=2, size=4, name='l')

    def test_create(self):
        self.assert_param_names(['w', 'b'])
        self.assert_count(12)

    def test_transform(self):
        out, upd = self.l.transform(dict(out=self.x))
        assert len(out) == 2
        assert not upd


class TestMaxout(Base):
    def _build(self):
        return theanets.layers.Maxout(inputs=2, size=4, pieces=3, name='l')

    def test_create(self):
        self.assert_param_names(['b', 'w'])
        self.assert_count(28)

    def test_transform(self):
        out, upd = self.l.transform(dict(out=self.x))
        assert len(out) == 2
        assert not upd


class TestRNN(Base):
    def _build(self):
        return theanets.layers.RNN(inputs=2, size=4, name='l')

    def test_create(self):
        self.assert_param_names(['b', 'hh', 'xh'])
        self.assert_count(28)

    def test_transform(self):
        out, upd = self.l.transform(dict(out=self.x))
        assert len(out) == 2
        assert not upd


class TestARRNN(Base):
    def _build(self):
        return theanets.layers.ARRNN(inputs=2, size=4, name='l')

    def test_create(self):
        self.assert_param_names(['b', 'hh', 'r', 'xh', 'xr'])
        self.assert_count(40)

    def test_transform(self):
        out, upd = self.l.transform(dict(out=self.x))
        assert len(out) == 4
        assert not upd


class TestLRRNN(Base):
    def _build(self):
        return theanets.layers.LRRNN(inputs=2, size=4, name='l')

    def test_create(self):
        self.assert_param_names(['b', 'hh', 'r', 'xh'])
        self.assert_count(32)

    def test_transform(self):
        out, upd = self.l.transform(dict(out=self.x))
        assert len(out) == 4
        assert not upd


class TestMRNN(Base):
    def _build(self):
        return theanets.layers.MRNN(inputs=2, size=4, factors=3, name='l')

    def test_create(self):
        self.assert_param_names(['b', 'fh', 'hf', 'xf', 'xh'])
        self.assert_count(42)

    def test_transform(self):
        out, upd = self.l.transform(dict(out=self.x))
        assert len(out) == 3
        assert not upd


class TestLSTM(Base):
    def _build(self):
        return theanets.layers.LSTM(inputs=2, size=4, name='l')

    def test_create(self):
        self.assert_param_names(['b', 'cf', 'ci', 'co', 'hh', 'xh'])
        self.assert_count(124)

    def test_transform(self):
        out, upd = self.l.transform(dict(out=self.x))
        assert len(out) == 2
        assert not upd


class TestGRU(Base):
    def _build(self):
        return theanets.layers.GRU(inputs=2, size=4, name='l')

    def test_create(self):
        self.assert_param_names(['b', 'xh', 'hh'])
        self.assert_count(84)

    def test_transform(self):
        out, upd = self.l.transform(dict(out=self.x))
        assert len(out) == 4
        assert not upd


class TestClockwork(Base):
    def _build(self):
        return theanets.layers.Clockwork(inputs=2, size=4, periods=(2, 5), name='l')

    def test_create(self):
        self.assert_param_names(['b', 'xh', 'hh2', 'hh5'])
        self.assert_count(24)

    def test_transform(self):
        out, upd = self.l.transform(dict(out=self.x))
        assert len(out) == 2
        assert not upd

    def test_spec(self):
        self.assert_spec(periods=(2, 5), size=4)

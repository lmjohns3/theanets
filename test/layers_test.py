import numpy as np
import theanets
import theano.tensor as TT


class Base(object):
    INPUTS = 2
    SIZE = 4
    OUTPUTS = 3

    def setUp(self):
        self.x = TT.matrix('x')
        self.l = self._build()

    def test_feed_forward(self):
        net = theanets.Regressor((Base.INPUTS, self.l, Base.OUTPUTS))
        out = net.predict(np.random.randn(8, Base.INPUTS).astype('f'))
        assert out.shape == (8, Base.OUTPUTS)

    def assert_param_names(self, expected):
        if not expected[0].startswith('l'):
            expected = sorted('l.{}'.format(n) for n in expected)
        real = sorted(p.name for p in self.l.params)
        assert real == expected, 'got {}, expected {}'.format(real, expected)

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
        return theanets.layers.Feedforward(
            inputs=Base.INPUTS, size=Base.SIZE, name='l')

    def test_build(self):
        for f in 'feedforward Feedforward classifier rnn lstm'.split():
            l = theanets.layers.build(f, inputs=Base.INPUTS, size=Base.SIZE)
            assert isinstance(l, theanets.layers.Layer)

    def test_connect(self):
        out, upd = self.l.connect(dict(out=self.x))
        assert len(out) == 2
        assert len(upd) == 0


class TestFeedforward(Base):
    def _build(self):
        return theanets.layers.Feedforward(
            inputs=Base.INPUTS, size=Base.SIZE, name='l')

    def test_create(self):
        self.assert_param_names(['w', 'b'])
        self.assert_count(12)

    def test_transform(self):
        out, upd = self.l.transform(dict(out=self.x))
        assert len(out) == 2
        assert not upd


class TestMultiFeedforward(Base):
    def _build(self):
        self.a = theanets.layers.Feedforward(
            inputs={'in:out': Base.INPUTS}, size=Base.SIZE, name='a')
        self.b = theanets.layers.Feedforward(
            inputs={'in:out': Base.INPUTS}, size=Base.SIZE, name='b')
        return theanets.layers.Feedforward(
            inputs={'a:out': Base.SIZE, 'b:out': Base.SIZE},
            size=Base.SIZE, name='l')

    def test_feed_forward(self):
        net = theanets.Regressor(
            (Base.INPUTS, self.a, self.b, self.l, Base.OUTPUTS))
        out = net.predict(np.random.randn(8, Base.INPUTS).astype('f'))
        assert out.shape == (8, Base.OUTPUTS)

    def test_create(self):
        self.assert_param_names(['w_a:out', 'w_b:out', 'b'])
        self.assert_count(36)

    def test_transform(self):
        out, upd = self.l.transform({'a:out': self.x, 'b:out': self.x})
        assert len(out) == 2
        assert not upd


class TestTied(Base):
    def _build(self):
        self.l0 = theanets.layers.Feedforward(
            inputs=Base.INPUTS, size=Base.SIZE, name='l0')
        return theanets.layers.Tied(partner=self.l0, name='l')

    def test_feed_forward(self):
        net = theanets.Autoencoder((Base.INPUTS, self.l0, self.l))
        out = net.predict(np.random.randn(8, Base.INPUTS).astype('f'))
        assert out.shape == (8, Base.INPUTS)

    def test_create(self):
        l = self._build()
        assert sorted(p.name for p in l.params) == [l.name + '.b']
        self.assert_count(2)

    def test_transform(self):
        out, upd = self.l.transform(dict(out=self.x))
        assert len(out) == 2
        assert not upd


class TestClassifier(Base):
    def _build(self):
        return theanets.layers.Classifier(
            inputs=Base.INPUTS, size=Base.SIZE, name='l')

    def test_create(self):
        self.assert_param_names(['w', 'b'])
        self.assert_count(12)

    def test_transform(self):
        out, upd = self.l.transform(dict(out=self.x))
        assert len(out) == 2
        assert not upd


class BaseRecurrent(Base):
    def setUp(self):
        super(BaseRecurrent, self).setUp()
        self.x = TT.tensor3('x')

    def test_feed_forward(self):
        net = theanets.recurrent.Regressor((Base.INPUTS, self.l, Base.OUTPUTS))
        out = net.predict(np.random.randn(8, 5, Base.INPUTS).astype('f'))
        assert out.shape == (8, 5, Base.OUTPUTS)


class TestConv1(BaseRecurrent):
    def _build(self):
        return theanets.layers.Conv1(
            inputs=Base.INPUTS, size=Base.SIZE, filter_size=3, name='l')

    def test_feed_forward(self):
        net = theanets.recurrent.Regressor((Base.INPUTS, self.l, Base.OUTPUTS))
        out = net.predict(np.random.randn(5, 8, Base.INPUTS).astype('f'))
        assert out.shape == (5, 6, Base.OUTPUTS)

    def test_create(self):
        self.assert_param_names(['b', 'w'])
        self.assert_count(28)

    def test_transform(self):
        out, upd = self.l.transform(dict(out=self.x))
        assert len(out) == 2
        assert not upd


class TestRNN(BaseRecurrent):
    def _build(self):
        return theanets.layers.RNN(
            inputs=Base.INPUTS, size=Base.SIZE, name='l')

    def test_create(self):
        self.assert_param_names(['b', 'hh', 'xh'])
        self.assert_count(28)

    def test_transform(self):
        out, upd = self.l.transform(dict(out=self.x))
        assert len(out) == 2
        assert not upd


class TestARRNN(BaseRecurrent):
    def _build(self):
        return theanets.layers.ARRNN(
            inputs=Base.INPUTS, size=Base.SIZE, name='l')

    def test_create(self):
        self.assert_param_names(['b', 'hh', 'r', 'xh', 'xr'])
        self.assert_count(40)

    def test_transform(self):
        out, upd = self.l.transform(dict(out=self.x))
        assert len(out) == 4
        assert not upd


class TestLRRNN(BaseRecurrent):
    def _build(self):
        return theanets.layers.LRRNN(
            inputs=Base.INPUTS, size=Base.SIZE, name='l')

    def test_create(self):
        self.assert_param_names(['b', 'hh', 'r', 'xh'])
        self.assert_count(32)

    def test_transform(self):
        out, upd = self.l.transform(dict(out=self.x))
        assert len(out) == 4
        assert not upd


class TestMRNN(BaseRecurrent):
    def _build(self):
        return theanets.layers.MRNN(
            inputs=Base.INPUTS, size=Base.SIZE, factors=3, name='l')

    def test_create(self):
        self.assert_param_names(['b', 'fh', 'hf', 'xf', 'xh'])
        self.assert_count(42)

    def test_transform(self):
        out, upd = self.l.transform(dict(out=self.x))
        assert len(out) == 3
        assert not upd


class TestLSTM(BaseRecurrent):
    def _build(self):
        return theanets.layers.LSTM(
            inputs=Base.INPUTS, size=Base.SIZE, name='l')

    def test_create(self):
        self.assert_param_names(['b', 'cf', 'ci', 'co', 'hh', 'xh'])
        self.assert_count(124)

    def test_transform(self):
        out, upd = self.l.transform(dict(out=self.x))
        assert len(out) == 2
        assert not upd


class TestGRU(BaseRecurrent):
    def _build(self):
        return theanets.layers.GRU(
            inputs=Base.INPUTS, size=Base.SIZE, name='l')

    def test_create(self):
        self.assert_param_names(['bh', 'br', 'bz',
                                 'hh', 'hr', 'hz',
                                 'xh', 'xr', 'xz'])
        self.assert_count(84)

    def test_transform(self):
        out, upd = self.l.transform(dict(out=self.x))
        assert len(out) == 4
        assert not upd


class TestClockwork(BaseRecurrent):
    def _build(self):
        return theanets.layers.Clockwork(
            inputs=Base.INPUTS, size=Base.SIZE, periods=(2, 5), name='l')

    def test_create(self):
        self.assert_param_names(['b', 'xh', 'hh'])
        self.assert_count(28)

    def test_transform(self):
        out, upd = self.l.transform(dict(out=self.x))
        assert len(out) == 2
        assert not upd

    def test_spec(self):
        self.assert_spec(periods=(5, 2), size=Base.SIZE, form='clockwork')


class TestBidirectional(BaseRecurrent):
    def _build(self):
        return theanets.layers.Bidirectional(
            inputs=Base.INPUTS, size=Base.SIZE, worker='arrnn', name='l')

    def test_create(self):
        self.assert_param_names(
            ['l_bw.b', 'l_bw.hh', 'l_bw.r', 'l_bw.xh', 'l_bw.xr',
             'l_fw.b', 'l_fw.hh', 'l_fw.r', 'l_fw.xh', 'l_fw.xr'])
        self.assert_count(32)

    def test_transform(self):
        out, upd = self.l.transform(dict(out=self.x))
        assert len(out) == 10, 'got {}'.format(out)
        assert not upd

    def test_spec(self):
        self.assert_spec(size=Base.SIZE, form='bidirectional', worker='arrnn')

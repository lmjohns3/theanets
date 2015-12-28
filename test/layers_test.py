import numpy as np
import theanets
import theano.tensor as TT

import util


class Base(util.Base):
    NUM_HIDDEN = 4  # this needs to be an even integer for concatenate test.

    def setUp(self):
        self.x = TT.matrix('x')
        self.l = self._build()

    def test_feed_forward(self):
        net = theanets.Regressor((self.NUM_INPUTS, self.l, self.NUM_OUTPUTS))
        out = net.predict(self.INPUTS)
        assert out.shape == (self.NUM_EXAMPLES, self.NUM_OUTPUTS)

    def assert_param_names(self, expected):
        if expected and not expected[0].startswith('l'):
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
            inputs=self.NUM_INPUTS, size=self.NUM_HIDDEN, name='l')

    def test_build(self):
        for f in 'feedforward Feedforward classifier prod rnn lstm'.split():
            l = theanets.Layer.build(f, inputs=self.NUM_INPUTS, size=self.NUM_HIDDEN)
            assert isinstance(l, theanets.layers.Layer)

    def test_connect(self):
        out, upd = self.l.connect(dict(out=self.x))
        assert len(out) == 2
        assert len(upd) == 0


class TestFeedforward(Base):
    def _build(self):
        return theanets.layers.Feedforward(
            inputs=self.NUM_INPUTS, size=self.NUM_HIDDEN, name='l')

    def test_create(self):
        self.assert_param_names(['w', 'b'])
        self.assert_count((1 + self.NUM_INPUTS) * self.NUM_HIDDEN)

    def test_transform(self):
        out, upd = self.l.transform(dict(out=self.x))
        assert len(out) == 2
        assert not upd


class TestMultiFeedforward(Base):
    def _build(self):
        self.a = theanets.layers.Feedforward(
            inputs={'in:out': self.NUM_INPUTS}, size=self.NUM_HIDDEN, name='a')
        self.b = theanets.layers.Feedforward(
            inputs={'in:out': self.NUM_INPUTS}, size=self.NUM_HIDDEN, name='b')
        return theanets.layers.Feedforward(
            inputs={'a:out': self.NUM_HIDDEN, 'b:out': self.NUM_HIDDEN},
            size=self.NUM_HIDDEN, name='l')

    def test_feed_forward(self):
        net = theanets.Regressor(
            (self.NUM_INPUTS, self.a, self.b, self.l, self.NUM_OUTPUTS))
        out = net.predict(self.INPUTS)
        assert out.shape == (self.NUM_EXAMPLES, self.NUM_OUTPUTS)

    def test_create(self):
        self.assert_param_names(['w_a:out', 'w_b:out', 'b'])
        self.assert_count((1 + 2 * self.NUM_HIDDEN) * self.NUM_HIDDEN)

    def test_transform(self):
        out, upd = self.l.transform({'a:out': self.x, 'b:out': self.x})
        assert len(out) == 2
        assert not upd


class TestProduct(Base):
    def _build(self):
        self.a = theanets.layers.Feedforward(
            inputs={'in:out': self.NUM_INPUTS}, size=self.NUM_HIDDEN, name='a')
        self.b = theanets.layers.Feedforward(
            inputs={'in:out': self.NUM_INPUTS}, size=self.NUM_HIDDEN, name='b')
        return theanets.layers.Product(
            inputs={'a:out': self.NUM_HIDDEN, 'b:out': self.NUM_HIDDEN},
            size=self.NUM_HIDDEN, name='l')

    def test_feed_forward(self):
        net = theanets.Regressor(
            (self.NUM_INPUTS, self.a, self.b, self.l, self.NUM_OUTPUTS))
        out = net.predict(self.INPUTS)
        assert out.shape == (self.NUM_EXAMPLES, self.NUM_OUTPUTS)

    def test_create(self):
        self.assert_param_names([])
        self.assert_count(0)

    def test_transform(self):
        out, upd = self.l.transform({'a:out': self.x, 'b:out': self.x})
        assert len(out) == 1
        assert not upd


class TestConcatenate(Base):
    def _build(self):
        self.a = theanets.layers.Feedforward(
            inputs={'in:out': self.NUM_INPUTS}, size=self.NUM_HIDDEN // 2, name='a')
        self.b = theanets.layers.Feedforward(
            inputs={'in:out': self.NUM_INPUTS}, size=self.NUM_HIDDEN // 2, name='b')
        return theanets.layers.Concatenate(
            inputs={'a:out': self.NUM_HIDDEN // 2, 'b:out': self.NUM_HIDDEN // 2},
            size=self.NUM_HIDDEN, name='l')

    def test_feed_forward(self):
        net = theanets.Regressor(
            (self.NUM_INPUTS, self.a, self.b, self.l, self.NUM_OUTPUTS))
        for l in net.layers:
            print(l.name, l.__class__, l.inputs, l.size)
        out = net.predict(self.INPUTS)
        assert out.shape == (self.NUM_EXAMPLES, self.NUM_OUTPUTS)

    def test_create(self):
        self.assert_param_names([])
        self.assert_count(0)

    def test_transform(self):
        out, upd = self.l.transform({'a:out': self.x, 'b:out': self.x})
        assert len(out) == 1
        assert not upd


class TestTied(Base):
    def _build(self):
        self.l0 = theanets.layers.Feedforward(
            inputs=self.NUM_INPUTS, size=self.NUM_HIDDEN, name='l0')
        return theanets.layers.Tied(partner=self.l0, name='l')

    def test_feed_forward(self):
        net = theanets.Autoencoder((self.NUM_INPUTS, self.l0, self.l))
        out = net.predict(self.INPUTS)
        assert out.shape == (self.NUM_EXAMPLES, self.NUM_INPUTS)

    def test_create(self):
        l = self._build()
        assert sorted(p.name for p in l.params) == [l.name + '.b']
        self.assert_count(self.NUM_INPUTS)

    def test_transform(self):
        out, upd = self.l.transform(dict(out=self.x))
        assert len(out) == 2
        assert not upd


class TestClassifier(Base):
    def _build(self):
        return theanets.layers.Classifier(
            inputs=self.NUM_INPUTS, size=self.NUM_CLASSES, name='l')

    def test_feed_forward(self):
        net = theanets.Classifier((self.NUM_INPUTS, self.l))
        out = net.predict_proba(self.INPUTS)
        assert out.shape == (self.NUM_EXAMPLES, self.NUM_CLASSES)

    def test_create(self):
        self.assert_param_names(['w', 'b'])
        self.assert_count((1 + self.NUM_INPUTS) * self.NUM_CLASSES)

    def test_transform(self):
        out, upd = self.l.transform(dict(out=self.x))
        assert len(out) == 2
        assert not upd


class BaseRecurrent(Base):
    NUM_TIMES = 12

    INPUTS = np.random.randn(Base.NUM_EXAMPLES, NUM_TIMES, Base.NUM_INPUTS).astype('f')

    def setUp(self):
        super(BaseRecurrent, self).setUp()
        self.x = TT.tensor3('x')

    def test_feed_forward(self):
        net = theanets.recurrent.Regressor((self.NUM_INPUTS, self.l, self.NUM_OUTPUTS))
        out = net.predict(self.INPUTS)
        assert out.shape == (self.NUM_EXAMPLES, self.NUM_TIMES, self.NUM_OUTPUTS)


class TestReshape(BaseRecurrent):
    def _build(self):
        # we get an input that's (BATCH, TIMES, INPUTS): (64, 12, 7).
        # we'll reshape to (BATCH, 14, 6): (64, 14, 6).
        #
        # of course this doesn't make any sense for a real nn to do, it's just
        # to test the reshaping operation.
        return theanets.layers.Reshape(inputs=self.NUM_INPUTS, shape=(14, 6))

    def test_create(self):
        self.assert_param_names([])
        self.assert_count(0)

    def test_transform(self):
        out, upd = self.l.transform(dict(out=self.x))
        assert len(out) == 1
        assert not upd

    def test_spec(self):
        self.assert_spec(shape=[14, 6], size=6, form='reshape')

    def test_feed_forward(self):
        net = theanets.recurrent.Regressor((self.NUM_INPUTS, self.l, self.NUM_OUTPUTS))
        out = net.predict(self.INPUTS)
        assert out.shape == (self.NUM_EXAMPLES, 14, self.NUM_OUTPUTS)


class TestFlatten(BaseRecurrent):
    def _build(self):
        # we get an input that's (BATCH, TIMES, INPUTS): (8, 12, 7).
        # this gets flattened to (BATCH, TIMES * INPUTS): (8, 84).
        return theanets.layers.Flatten(inputs=self.NUM_INPUTS, size=84)

    def test_create(self):
        self.assert_param_names([])
        self.assert_count(0)

    def test_transform(self):
        out, upd = self.l.transform(dict(out=self.x))
        assert len(out) == 1
        assert not upd

    def test_feed_forward(self):
        net = theanets.recurrent.Regressor((self.NUM_INPUTS, self.l, self.NUM_OUTPUTS))
        out = net.predict(self.INPUTS)
        assert out.shape == (self.NUM_EXAMPLES, self.NUM_OUTPUTS)


class TestConv1(BaseRecurrent):
    def _build(self):
        return theanets.layers.Conv1(
            inputs=self.NUM_INPUTS, size=self.NUM_HIDDEN, filter_size=3, name='l')

    def test_feed_forward(self):
        net = theanets.recurrent.Regressor((self.NUM_INPUTS, self.l, self.NUM_OUTPUTS))
        out = net.predict(self.INPUTS)
        assert out.shape == (self.NUM_EXAMPLES, self.NUM_TIMES - 2, self.NUM_OUTPUTS)

    def test_create(self):
        self.assert_param_names(['b', 'w'])
        self.assert_count((1 + 3 * self.NUM_INPUTS) * self.NUM_HIDDEN)

    def test_transform(self):
        out, upd = self.l.transform(dict(out=self.x))
        assert len(out) == 2
        assert not upd


class TestRNN(BaseRecurrent):
    def _build(self):
        return theanets.layers.RNN(
            inputs=self.NUM_INPUTS, size=self.NUM_HIDDEN, name='l')

    def test_create(self):
        self.assert_param_names(['b', 'hh', 'xh'])
        self.assert_count((1 + self.NUM_HIDDEN + self.NUM_INPUTS) * self.NUM_HIDDEN)

    def test_transform(self):
        out, upd = self.l.transform(dict(out=self.x))
        assert len(out) == 2
        assert not upd


class TestARRNN(BaseRecurrent):
    def _build(self):
        return theanets.layers.RRNN(
            inputs=self.NUM_INPUTS, size=self.NUM_HIDDEN, name='l')

    def test_create(self):
        self.assert_param_names(['b', 'hh', 'r', 'xh', 'xr'])
        self.assert_count(
            (1 + 1 + self.NUM_HIDDEN + 2 * self.NUM_INPUTS) * self.NUM_HIDDEN)

    def test_transform(self):
        out, upd = self.l.transform(dict(out=self.x))
        assert len(out) == 4
        assert not upd


class TestLRRNN(BaseRecurrent):
    def _build(self):
        return theanets.layers.RRNN(
            inputs=self.NUM_INPUTS, size=self.NUM_HIDDEN, name='l',
            rate='vector')

    def test_create(self):
        self.assert_param_names(['b', 'hh', 'r', 'xh'])
        self.assert_count(
            (1 + 1 + self.NUM_INPUTS + self.NUM_HIDDEN) * self.NUM_HIDDEN)

    def test_transform(self):
        out, upd = self.l.transform(dict(out=self.x))
        assert len(out) == 4
        assert not upd


class TestRRNN(BaseRecurrent):
    def _build(self):
        return theanets.layers.RRNN(
            inputs=self.NUM_INPUTS, size=self.NUM_HIDDEN, name='l',
            rate='uniform')

    def test_create(self):
        self.assert_param_names(['b', 'hh', 'xh'])
        self.assert_count(
            (1 + self.NUM_INPUTS + self.NUM_HIDDEN) * self.NUM_HIDDEN)

    def test_transform(self):
        out, upd = self.l.transform(dict(out=self.x))
        assert len(out) == 4
        assert not upd


class TestMRNN(BaseRecurrent):
    def _build(self):
        return theanets.layers.MRNN(
            inputs=self.NUM_INPUTS, size=self.NUM_HIDDEN, factors=3, name='l')

    def test_create(self):
        self.assert_param_names(['b', 'fh', 'hf', 'xf', 'xh'])
        self.assert_count(
            (1 + 3 + 3 + self.NUM_INPUTS) * self.NUM_HIDDEN + 3 * self.NUM_INPUTS)

    def test_transform(self):
        out, upd = self.l.transform(dict(out=self.x))
        assert len(out) == 3
        assert not upd


class TestLSTM(BaseRecurrent):
    def _build(self):
        return theanets.layers.LSTM(
            inputs=self.NUM_INPUTS, size=self.NUM_HIDDEN, name='l')

    def test_create(self):
        self.assert_param_names(['b', 'cf', 'ci', 'co', 'hh', 'xh'])
        self.assert_count(
            (4 + 3 + 4 * self.NUM_HIDDEN + 4 * self.NUM_INPUTS) * self.NUM_HIDDEN)

    def test_transform(self):
        out, upd = self.l.transform(dict(out=self.x))
        assert len(out) == 2
        assert not upd


class TestGRU(BaseRecurrent):
    def _build(self):
        return theanets.layers.GRU(
            inputs=self.NUM_INPUTS, size=self.NUM_HIDDEN, name='l')

    def test_create(self):
        self.assert_param_names(['bh', 'br', 'bz',
                                 'hh', 'hr', 'hz',
                                 'xh', 'xr', 'xz'])
        self.assert_count(
            3 * (1 + self.NUM_INPUTS + self.NUM_HIDDEN) * self.NUM_HIDDEN)

    def test_transform(self):
        out, upd = self.l.transform(dict(out=self.x))
        assert len(out) == 4
        assert not upd


class TestMUT1(BaseRecurrent):
    def _build(self):
        return theanets.layers.MUT1(
            inputs=self.NUM_INPUTS, size=self.NUM_HIDDEN, name='l')

    def test_create(self):
        self.assert_param_names(['bh', 'br', 'bz',
                                 'hh', 'hr',
                                 'xh', 'xr', 'xz'])
        self.assert_count(
            (3 + 3 * self.NUM_INPUTS + 2 * self.NUM_HIDDEN) * self.NUM_HIDDEN)

    def test_transform(self):
        out, upd = self.l.transform(dict(out=self.x))
        assert len(out) == 4
        assert not upd


class TestClockwork(BaseRecurrent):
    def _build(self):
        return theanets.layers.Clockwork(
            inputs=self.NUM_INPUTS, size=self.NUM_HIDDEN, periods=(2, 5), name='l')

    def test_create(self):
        self.assert_param_names(['b', 'xh', 'hh'])
        self.assert_count((1 + self.NUM_INPUTS + self.NUM_HIDDEN) * self.NUM_HIDDEN)

    def test_transform(self):
        out, upd = self.l.transform(dict(out=self.x))
        assert len(out) == 2
        assert not upd

    def test_spec(self):
        self.assert_spec(periods=(5, 2), size=self.NUM_HIDDEN, form='clockwork')


class TestSCRN(BaseRecurrent):
    def _build(self):
        return theanets.layers.SCRN(
            inputs=self.NUM_INPUTS, size=self.NUM_HIDDEN, name='l')

    def test_create(self):
        self.assert_param_names(['b', 'xh', 'hh', 'sh', 'xs', 'ho', 'so', 'r'])
        self.assert_count(
            2 * (1 + self.NUM_INPUTS + 2 * self.NUM_HIDDEN) * self.NUM_HIDDEN)

    def test_transform(self):
        out, upd = self.l.transform(dict(out=self.x))
        assert len(out) == 5
        assert not upd

    def test_spec(self):
        self.assert_spec(size=self.NUM_HIDDEN, form='scrn')


class TestBidirectional(BaseRecurrent):
    def _build(self):
        return theanets.layers.Bidirectional(
            inputs=self.NUM_INPUTS, size=self.NUM_HIDDEN, worker='rrnn', name='l')

    def test_create(self):
        self.assert_param_names(
            ['l_bw.b', 'l_bw.hh', 'l_bw.r', 'l_bw.xh', 'l_bw.xr',
             'l_fw.b', 'l_fw.hh', 'l_fw.r', 'l_fw.xh', 'l_fw.xr'])
        h = self.NUM_HIDDEN // 2
        self.assert_count(2 * h * (2 + h + 2 * self.NUM_INPUTS))

    def test_transform(self):
        out, upd = self.l.transform(dict(out=self.x))
        assert len(out) == 10, 'got {}'.format(out)
        assert not upd

    def test_spec(self):
        self.assert_spec(size=self.NUM_HIDDEN, form='bidirectional', worker='rrnn')

import numpy as np
import pytest
import theanets
import theano.tensor as TT

import util


class Base(util.Base):
    NUM_HIDDEN = 4  # this needs to be an even integer for concatenate test.

    @pytest.fixture
    def x(self):
        return TT.matrix('x')

    @pytest.fixture
    def ffa(self):
        return theanets.layers.Feedforward(
            inputs={'in:out': self.NUM_INPUTS}, size=self.NUM_HIDDEN, name='a')

    @pytest.fixture
    def ffb(self):
        return theanets.layers.Feedforward(
            inputs={'in:out': self.NUM_INPUTS}, size=self.NUM_HIDDEN, name='b')

    def test_feed_forward(self, layer):
        net = theanets.Regressor((self.NUM_INPUTS, layer, self.NUM_OUTPUTS))
        out = net.predict(self.INPUTS)
        assert out.shape == (self.NUM_EXAMPLES, self.NUM_OUTPUTS)

    def assert_param_names(self, layer, expected):
        if expected and not expected[0].startswith('l'):
            expected = sorted('l.{}'.format(n) for n in expected)
        assert sorted(p.name for p in layer.params) == expected

    def assert_count(self, layer, expected):
        assert layer.num_params == expected

    def assert_spec(self, layer, **expected):
        assert layer.to_spec() == expected
        return
        for k, v in expected.items():
            r = real.get(k)
            assert r is not None
            if isinstance(v, np.ndarray) or isinstance(r, np.ndarray):
                assert np.allclose(r, v)
            else:
                assert r == v


class TestLayer(Base):
    @pytest.fixture
    def layer(self):
        return theanets.layers.Feedforward(
            inputs=self.NUM_INPUTS, size=self.NUM_HIDDEN, name='l')

    def test_build(self):
        for f in 'feedforward Feedforward classifier prod rnn lstm'.split():
            l = theanets.Layer.build(f, inputs=self.NUM_INPUTS, size=self.NUM_HIDDEN)
            assert isinstance(l, theanets.layers.Layer)

    def test_connect(self, layer, x):
        out, upd = layer.connect(dict(out=x))
        assert len(out) == 2
        assert len(upd) == 0


class TestFeedforward(Base):
    @pytest.fixture
    def layer(self):
        return theanets.layers.Feedforward(
            inputs=self.NUM_INPUTS, size=self.NUM_HIDDEN, name='l')

    def test_create(self, layer):
        self.assert_param_names(layer, ['w', 'b'])
        self.assert_count(layer, (1 + self.NUM_INPUTS) * self.NUM_HIDDEN)

    def test_transform(self, layer, x):
        out, upd = layer.transform(dict(out=x))
        assert len(out) == 2
        assert not upd


class TestMultiFeedforward(Base):
    @pytest.fixture
    def layer(self):
        return theanets.layers.Feedforward(
            inputs={'a:out': self.NUM_HIDDEN, 'b:out': self.NUM_HIDDEN},
            size=self.NUM_HIDDEN, name='l')

    def test_feed_forward(self, layer, ffa, ffb):
        net = theanets.Regressor((self.NUM_INPUTS, ffa, ffb, layer, self.NUM_OUTPUTS))
        out = net.predict(self.INPUTS)
        assert out.shape == (self.NUM_EXAMPLES, self.NUM_OUTPUTS)

    def test_create(self, layer):
        self.assert_param_names(layer, ['w_a:out', 'w_b:out', 'b'])
        self.assert_count(layer, (1 + 2 * self.NUM_HIDDEN) * self.NUM_HIDDEN)

    def test_transform(self, layer, x):
        out, upd = layer.transform({'a:out': x, 'b:out': x})
        assert len(out) == 2
        assert not upd


class TestProduct(Base):
    @pytest.fixture
    def layer(self):
        return theanets.layers.Product(
            inputs={'a:out': self.NUM_HIDDEN, 'b:out': self.NUM_HIDDEN},
            size=self.NUM_HIDDEN, name='l')

    def test_feed_forward(self, layer, ffa, ffb):
        net = theanets.Regressor((self.NUM_INPUTS, ffa, ffb, layer, self.NUM_OUTPUTS))
        out = net.predict(self.INPUTS)
        assert out.shape == (self.NUM_EXAMPLES, self.NUM_OUTPUTS)

    def test_create(self, layer):
        self.assert_param_names(layer, [])
        self.assert_count(layer, 0)

    def test_transform(self, layer, x):
        out, upd = layer.transform({'a:out': x, 'b:out': x})
        assert len(out) == 1
        assert not upd


class TestConcatenate(Base):
    @pytest.fixture
    def layer(self):
        return theanets.layers.Concatenate(
            inputs={'a:out': self.NUM_HIDDEN, 'b:out': self.NUM_HIDDEN},
            size=2 * self.NUM_HIDDEN, name='l')

    def test_feed_forward(self, layer, ffa, ffb):
        net = theanets.Regressor((self.NUM_INPUTS, ffa, ffb, layer, self.NUM_OUTPUTS))
        for l in net.layers:
            print(l.name, l.__class__, l.inputs, l.size)
        out = net.predict(self.INPUTS)
        assert out.shape == (self.NUM_EXAMPLES, self.NUM_OUTPUTS)

    def test_create(self, layer):
        self.assert_param_names(layer, [])
        self.assert_count(layer, 0)

    def test_transform(self, layer, x):
        out, upd = layer.transform({'a:out': x, 'b:out': x})
        assert len(out) == 1
        assert not upd


class TestTied(Base):
    @pytest.fixture
    def l0(self):
        return theanets.layers.Feedforward(
            inputs=self.NUM_INPUTS, size=self.NUM_HIDDEN, name='l0')

    @pytest.fixture
    def layer(self, l0):
        return theanets.layers.Tied(partner=l0, name='l')

    def test_feed_forward(self, layer, l0):
        net = theanets.Autoencoder((self.NUM_INPUTS, l0, layer))
        out = net.predict(self.INPUTS)
        assert out.shape == (self.NUM_EXAMPLES, self.NUM_INPUTS)

    def test_create(self, layer):
        assert sorted(p.name for p in layer.params) == [layer.name + '.b']
        self.assert_count(layer, self.NUM_INPUTS)

    def test_transform(self, layer, x):
        out, upd = layer.transform(dict(out=x))
        assert len(out) == 2
        assert not upd


class TestClassifier(Base):
    @pytest.fixture
    def layer(self):
        return theanets.layers.Classifier(
            inputs=self.NUM_INPUTS, size=self.NUM_CLASSES, name='l')

    def test_feed_forward(self, layer):
        net = theanets.Classifier((self.NUM_INPUTS, layer))
        out = net.predict_proba(self.INPUTS)
        assert out.shape == (self.NUM_EXAMPLES, self.NUM_CLASSES)

    def test_create(self, layer):
        self.assert_param_names(layer, ['w', 'b'])
        self.assert_count(layer, (1 + self.NUM_INPUTS) * self.NUM_CLASSES)

    def test_transform(self, layer, x):
        out, upd = layer.transform(dict(out=x))
        assert len(out) == 2
        assert not upd


class BaseRecurrent(Base):
    NUM_TIMES = 12

    INPUTS = np.random.randn(Base.NUM_EXAMPLES, NUM_TIMES, Base.NUM_INPUTS).astype('f')

    @pytest.fixture
    def x(self):
        return TT.tensor3('x')

    def test_feed_forward(self, layer):
        net = theanets.recurrent.Regressor((self.NUM_INPUTS, layer, self.NUM_OUTPUTS))
        out = net.predict(self.INPUTS)
        assert out.shape == (self.NUM_EXAMPLES, self.NUM_TIMES, self.NUM_OUTPUTS)


class TestReshape(BaseRecurrent):
    @pytest.fixture
    def layer(self):
        # we get an input that's (BATCH, TIMES, INPUTS): (64, 12, 7).
        # we'll reshape to (BATCH, 14, 6): (64, 14, 6).
        #
        # of course this doesn't make any sense for a real nn to do, it's just
        # to test the reshaping operation.
        return theanets.layers.Reshape(inputs=self.NUM_INPUTS, shape=(14, 6), name='l')

    def test_create(self, layer):
        self.assert_param_names(layer, [])
        self.assert_count(layer, 0)

    def test_transform(self, layer, x):
        out, upd = layer.transform(dict(out=x))
        assert len(out) == 1
        assert not upd

    def test_spec(self, layer):
        self.assert_spec(layer, shape=[14, 6], size=6, form='reshape',
                         activation='relu', inputs={'out': 7}, name='l')

    def test_feed_forward(self, layer):
        net = theanets.recurrent.Regressor((self.NUM_INPUTS, layer, self.NUM_OUTPUTS))
        out = net.predict(self.INPUTS)
        assert out.shape == (self.NUM_EXAMPLES, 14, self.NUM_OUTPUTS)


class TestFlatten(BaseRecurrent):
    @pytest.fixture
    def layer(self):
        # we get an input that's (BATCH, TIMES, INPUTS): (8, 12, 7).
        # this gets flattened to (BATCH, TIMES * INPUTS): (8, 84).
        return theanets.layers.Flatten(inputs=self.NUM_INPUTS, size=84)

    def test_create(self, layer):
        self.assert_param_names(layer, [])
        self.assert_count(layer, 0)

    def test_transform(self, layer, x):
        out, upd = layer.transform(dict(out=x))
        assert len(out) == 1
        assert not upd

    def test_feed_forward(self, layer):
        net = theanets.recurrent.Regressor((self.NUM_INPUTS, layer, self.NUM_OUTPUTS))
        out = net.predict(self.INPUTS)
        assert out.shape == (self.NUM_EXAMPLES, self.NUM_OUTPUTS)


class TestConv1(BaseRecurrent):
    @pytest.fixture
    def layer(self):
        return theanets.layers.Conv1(
            inputs=self.NUM_INPUTS, size=self.NUM_HIDDEN, filter_size=3, name='l')

    def test_feed_forward(self, layer):
        net = theanets.recurrent.Regressor((self.NUM_INPUTS, layer, self.NUM_OUTPUTS))
        out = net.predict(self.INPUTS)
        assert out.shape == (self.NUM_EXAMPLES, self.NUM_TIMES - 2, self.NUM_OUTPUTS)

    def test_create(self, layer):
        self.assert_param_names(layer, ['b', 'w'])
        self.assert_count(layer, (1 + 3 * self.NUM_INPUTS) * self.NUM_HIDDEN)

    def test_transform(self, layer, x):
        out, upd = layer.transform(dict(out=x))
        assert len(out) == 2
        assert not upd


class TestRNN(BaseRecurrent):
    @pytest.fixture
    def layer(self):
        return theanets.layers.RNN(
            inputs=self.NUM_INPUTS, size=self.NUM_HIDDEN, name='l')

    def test_create(self, layer):
        self.assert_param_names(layer, ['b', 'hh', 'xh'])
        self.assert_count(
            layer, (1 + self.NUM_HIDDEN + self.NUM_INPUTS) * self.NUM_HIDDEN)

    def test_transform(self, layer, x):
        out, upd = layer.transform(dict(out=x))
        assert len(out) == 2
        assert not upd


class TestARRNN(BaseRecurrent):
    @pytest.fixture
    def layer(self):
        return theanets.layers.RRNN(
            inputs=self.NUM_INPUTS, size=self.NUM_HIDDEN, name='l')

    def test_create(self, layer):
        self.assert_param_names(layer, ['b', 'hh', 'r', 'xh', 'xr'])
        self.assert_count(
            layer, (1 + 1 + self.NUM_HIDDEN + 2 * self.NUM_INPUTS) * self.NUM_HIDDEN)

    def test_transform(self, layer, x):
        out, upd = layer.transform(dict(out=x))
        assert len(out) == 4
        assert not upd


class TestLRRNN(BaseRecurrent):
    @pytest.fixture
    def layer(self):
        return theanets.layers.RRNN(
            inputs=self.NUM_INPUTS, size=self.NUM_HIDDEN, name='l',
            rate='vector')

    def test_create(self, layer):
        self.assert_param_names(layer, ['b', 'hh', 'r', 'xh'])
        self.assert_count(
            layer, (1 + 1 + self.NUM_INPUTS + self.NUM_HIDDEN) * self.NUM_HIDDEN)

    def test_transform(self, layer, x):
        out, upd = layer.transform(dict(out=x))
        assert len(out) == 4
        assert not upd


class TestRRNN(BaseRecurrent):
    @pytest.fixture
    def layer(self):
        return theanets.layers.RRNN(
            inputs=self.NUM_INPUTS, size=self.NUM_HIDDEN, name='l',
            rate='uniform')

    def test_create(self, layer):
        self.assert_param_names(layer, ['b', 'hh', 'xh'])
        self.assert_count(
            layer, (1 + self.NUM_INPUTS + self.NUM_HIDDEN) * self.NUM_HIDDEN)

    def test_transform(self, layer, x):
        out, upd = layer.transform(dict(out=x))
        assert len(out) == 4
        assert not upd


class TestMRNN(BaseRecurrent):
    @pytest.fixture
    def layer(self):
        return theanets.layers.MRNN(
            inputs=self.NUM_INPUTS, size=self.NUM_HIDDEN, factors=3, name='l')

    def test_create(self, layer):
        self.assert_param_names(layer, ['b', 'fh', 'hf', 'xf', 'xh'])
        self.assert_count(
            layer,
            (1 + 3 + 3 + self.NUM_INPUTS) * self.NUM_HIDDEN + 3 * self.NUM_INPUTS)

    def test_transform(self, layer, x):
        out, upd = layer.transform(dict(out=x))
        assert len(out) == 3
        assert not upd

    def test_spec(self, layer):
        self.assert_spec(layer, name='l', form='mrnn', activation='relu',
                         factors=3, size=4, inputs=dict(out=7))


class TestLSTM(BaseRecurrent):
    @pytest.fixture
    def layer(self):
        return theanets.layers.LSTM(
            inputs=self.NUM_INPUTS, size=self.NUM_HIDDEN, name='l')

    def test_create(self, layer):
        self.assert_param_names(layer, ['b', 'cf', 'ci', 'co', 'hh', 'xh'])
        self.assert_count(
            layer,
            (4 + 3 + 4 * self.NUM_HIDDEN + 4 * self.NUM_INPUTS) * self.NUM_HIDDEN)

    def test_transform(self, layer, x):
        out, upd = layer.transform(dict(out=x))
        assert len(out) == 2
        assert not upd


class TestGRU(BaseRecurrent):
    @pytest.fixture
    def layer(self):
        return theanets.layers.GRU(
            inputs=self.NUM_INPUTS, size=self.NUM_HIDDEN, name='l')

    def test_create(self, layer):
        self.assert_param_names(layer, ['bh', 'br', 'bz',
                                        'hh', 'hr', 'hz',
                                        'xh', 'xr', 'xz'])
        self.assert_count(
            layer, 3 * (1 + self.NUM_INPUTS + self.NUM_HIDDEN) * self.NUM_HIDDEN)

    def test_transform(self, layer, x):
        out, upd = layer.transform(dict(out=x))
        assert len(out) == 4
        assert not upd


class TestMUT1(BaseRecurrent):
    @pytest.fixture
    def layer(self):
        return theanets.layers.MUT1(
            inputs=self.NUM_INPUTS, size=self.NUM_HIDDEN, name='l')

    def test_create(self, layer):
        self.assert_param_names(layer, ['bh', 'br', 'bz',
                                        'hh', 'hr',
                                        'xh', 'xr', 'xz'])
        self.assert_count(
            layer, (3 + 3 * self.NUM_INPUTS + 2 * self.NUM_HIDDEN) * self.NUM_HIDDEN)

    def test_transform(self, layer, x):
        out, upd = layer.transform(dict(out=x))
        assert len(out) == 4
        assert not upd


class TestClockwork(BaseRecurrent):
    @pytest.fixture
    def layer(self):
        return theanets.layers.Clockwork(
            inputs=self.NUM_INPUTS, size=self.NUM_HIDDEN, periods=(2, 5), name='l')

    def test_create(self, layer):
        self.assert_param_names(layer, ['b', 'xh', 'hh'])
        self.assert_count(
            layer, (1 + self.NUM_INPUTS + self.NUM_HIDDEN) * self.NUM_HIDDEN)

    def test_transform(self, layer, x):
        out, upd = layer.transform(dict(out=x))
        assert len(out) == 2
        assert not upd

    def test_spec(self, layer):
        self.assert_spec(layer, periods=(5, 2), size=self.NUM_HIDDEN, form='clockwork',
                         activation='relu', inputs={'out': 7}, name='l')


class TestSCRN(BaseRecurrent):
    @pytest.fixture
    def layer(self):
        return theanets.layers.SCRN(
            inputs=self.NUM_INPUTS, size=self.NUM_HIDDEN, name='l')

    def test_create(self, layer):
        self.assert_param_names(layer, ['b', 'xh', 'hh', 'sh', 'xs', 'ho', 'so', 'r'])
        self.assert_count(
            layer, 2 * (1 + self.NUM_INPUTS + 2 * self.NUM_HIDDEN) * self.NUM_HIDDEN)

    def test_transform(self, layer, x):
        out, upd = layer.transform(dict(out=x))
        assert len(out) == 5
        assert not upd

    def test_spec(self, layer):
        self.assert_spec(layer, size=self.NUM_HIDDEN, form='scrn', activation='relu',
                         inputs={'out': 7}, name='l')


class TestBidirectional(BaseRecurrent):
    @pytest.fixture
    def layer(self):
        return theanets.layers.Bidirectional(
            inputs=self.NUM_INPUTS, size=self.NUM_HIDDEN, worker='rrnn', name='l')

    def test_create(self, layer):
        self.assert_param_names(
            layer, ['l_bw.b', 'l_bw.hh', 'l_bw.r', 'l_bw.xh', 'l_bw.xr',
                    'l_fw.b', 'l_fw.hh', 'l_fw.r', 'l_fw.xh', 'l_fw.xr'])
        h = self.NUM_HIDDEN // 2
        self.assert_count(layer, 2 * h * (2 + h + 2 * self.NUM_INPUTS))

    def test_transform(self, layer, x):
        out, upd = layer.transform(dict(out=x))
        assert len(out) == 10
        assert not upd

    def test_spec(self, layer):
        self.assert_spec(layer, size=self.NUM_HIDDEN, form='bidirectional',
                         worker='rrnn', activation='relu', inputs={'out': 7}, name='l')

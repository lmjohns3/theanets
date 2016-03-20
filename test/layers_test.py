import numpy as np
import pytest
import theanets
import theano.tensor as TT

import util as u

NI = u.NUM_INPUTS
NH = u.NUM_HID1


class TestFeedforward:
    @pytest.mark.parametrize('form, name, params, count, outputs', [
        ('feedforward', 'feedforward', 'w b', 1 + NI, 'out pre'),
        ('ff', 'feedforward', 'w b', 1 + NI, 'out pre'),
        ('classifier', 'classifier', 'w b', 1 + NI, 'out pre'),
        ('flatten', 'flatten', '', 0, 'out'),
        ('flat', 'flatten', '', 0, 'out'),
        ('concatenate', 'concatenate', '', 0, 'out'),
        ('concat', 'concatenate', '', 0, 'out'),
        ('product', 'product', '', 0, 'out'),
        ('prod', 'product', '', 0, 'out'),
    ])
    def test_build(self, form, name, params, count, outputs):
        layer = theanets.Layer.build(form, size=NH, name='l', inputs='in')
        layer.bind(theanets.Network([NI]))

        assert layer.__class__.__name__.lower() == name

        assert sorted(p.name for p in layer.params) == \
            sorted('l.' + p for p in params.split())

        assert sum(np.prod(p.get_value().shape) for p in layer.params) == count * NH

        out, upd = layer.connect({'in:out': TT.matrix('x')})
        assert sorted(out) == sorted('l:' + o for o in outputs.split())
        assert sorted(upd) == []

        assert layer.to_spec() == dict(
            form=name, name='l', size=NH, inputs=('in:out', ),
            activation=layer.kwargs.get('activation', 'relu'))

    @pytest.mark.parametrize('layer', [
        NH,
        dict(form='ff', inputs=('hid1', 'hid2'), size=NH),
        dict(form='tied', partner='hid1'),
        dict(form='prod', inputs=('hid1', 'hid2'), size=NH),
        dict(form='concat', inputs=('hid1', 'hid2'), size=2 * NH),
        ('flat', NH),
    ])
    def test_predict(self, layer):
        net = theanets.Autoencoder([NI, NH, NH, layer, NI])
        assert net.predict(u.INPUTS).shape == (u.NUM_EXAMPLES, NI)

    def test_multiple_inputs(self):
        layer = theanets.layers.Feedforward(inputs=('in', 'hid1'), size=NH, name='l')
        layer.bind(theanets.Network([NH, NH, NH]))

        total = sum(np.prod(p.get_value().shape) for p in layer.params)
        assert total == (1 + 2 * NH) * NH

        assert sorted(p.name for p in layer.params) == \
            ['l.b', 'l.w_hid1:out', 'l.w_in:out']

        assert layer.to_spec() == dict(
            form='feedforward', name='l', size=NH, activation='relu',
            inputs=('in:out', 'hid1:out'))

    def test_reshape(self):
        layer = theanets.layers.Reshape(inputs='in', shape=(NI // 2, 2), name='l')
        layer.bind(theanets.Network([NI]))

        assert sum(np.prod(p.get_value().shape) for p in layer.params) == 0

        assert sorted(p.name for p in layer.params) == []

        assert layer.to_spec() == dict(
            form='reshape', name='l', size=2, shape=[NI // 2, 2],
            inputs=('in:out', ), activation='relu')


class TestRecurrent:
    @pytest.mark.parametrize('form, kwargs, count, params, outputs', [
        ('rnn', {}, 1 + NI + NH, 'xh hh b', 'out pre'),
        ('clockwork', {'periods': (1, 2, 4, 8)}, 1 + NI + NH, 'xh hh b', 'out pre'),
        ('rrnn', {'rate': 'uniform'}, 1 + NI + NH, 'xh hh b', 'out pre rate hid'),
        ('rrnn', {'rate': 'log'}, 1 + NI + NH, 'xh hh b', 'out pre rate hid'),
        ('rrnn', {'rate': 'vector'}, 2 + NI + NH, 'xh hh b r', 'out pre rate hid'),
        ('rrnn', {'rate': 'matrix'}, 2 + NH + 2 * NI, 'xh hh b r xr', 'out pre rate hid'),
        ('gru', {}, 3 * (1 + NI + NH), 'bh br bz xh xr xz hh hr hz', 'hid out pre rate'),
        ('mut1', {}, 3 + 3 * NI + 2 * NH, 'bh br bz hh hr xh xr xz', 'hid out pre rate'),
        ('lstm', {}, 7 + 4 * NH + 4 * NI, 'xh hh b cf ci co', 'out cell'),
        ('conv1', {'filter_size': 13}, 1 + 13 * NI, 'w b', 'pre out'),
        ('mrnn', {'factors': 3}, (7 + NI) * NH + 3 * NI, 'xh xf hf fh b',
         'out pre factors'),
        ('scrn', {}, 2 * (1 + NI + 2 * NH), 'xh xs ho so hh sh b r',
         'out pre hid rate state'),
        ('bidirectional', {}, 1 + NI + NH // 2,
         'l_bw.b l_bw.hh l_bw.xh l_fw.b l_fw.xh l_fw.hh',
         'bw_out bw_pre fw_out fw_pre out pre'),
    ])
    def test_build(self, form, kwargs, count, params, outputs):
        layer = theanets.Layer.build(form, size=NH, name='l', inputs='in', **kwargs)
        layer.bind(theanets.Network([dict(size=NI, ndim=3)]))

        assert layer.__class__.__name__.lower() == form

        expected = sorted('l.' + p for p in params.split())
        if form == 'bidirectional':
            expected = sorted(params.split())
        assert sorted(p.name for p in layer.params) == expected

        expected = count * NH
        if form == 'mrnn':
            expected = count
        assert sum(np.prod(p.get_value().shape) for p in layer.params) == expected

        out, upd = layer.connect({'in:out': TT.tensor3('x')})
        assert sorted(out) == sorted('l:' + o for o in outputs.split())
        assert sorted(upd) == []

        spec = {}
        if form == 'mrnn':
            spec['factors'] = 3
        if form == 'bidirectional':
            spec['worker'] = 'rnn'
        if form == 'clockwork':
            spec['periods'] = (1, 2, 4, 8)
        if form == 'scrn':
            spec['s_0'] = None
        if form == 'lstm':
            spec['c_0'] = None
        if form not in ('bidirectional', 'conv1'):
            spec['h_0'] = None
        assert layer.to_spec() == dict(
            form=form, name='l', size=NH, inputs=('in:out', ),
            activation=layer.kwargs.get('activation', 'relu'), **spec)

    @pytest.mark.parametrize('layer', [
        (NH, 'rnn'),
        dict(size=NH, form='conv1', filter_size=13),
    ])
    def test_predict(self, layer):
        T = u.RNN.NUM_TIMES
        if isinstance(layer, dict) and layer.get('form') == 'conv1':
            T -= layer['filter_size'] - 1
        net = theanets.recurrent.Autoencoder([NI, NH, NH, layer, NI])
        assert net.predict(u.RNN.INPUTS).shape == (u.NUM_EXAMPLES, T, NI)


class TestConvolution:
    @pytest.mark.parametrize('form, kwargs, count, params, outputs', [
        ('conv2', {'filter_size': u.CNN.FILTER_SIZE},
         1 + NI * u.CNN.FILTER_HEIGHT * u.CNN.FILTER_WIDTH, 'w b', 'out pre'),
    ])
    def test_build(self, form, kwargs, count, params, outputs):
        layer = theanets.Layer.build(form, size=NH, name='l', inputs='in', **kwargs)
        layer.bind(theanets.Network([dict(size=NI, ndim=4)]))

        assert layer.__class__.__name__.lower() == form

        expected = sorted('l.' + p for p in params.split())
        assert sorted(p.name for p in layer.params) == expected

        expected = count * NH
        assert sum(np.prod(p.get_value().shape) for p in layer.params) == expected

        out, upd = layer.connect({'in:out': TT.tensor4('x')})
        assert sorted(out) == sorted('l:' + o for o in outputs.split())
        assert sorted(upd) == []

        assert layer.to_spec() == dict(
            form=form, name='l', size=NH, inputs=('in:out', ),
            activation='relu')

    @pytest.mark.parametrize('layer', [
        dict(size=NH, form='conv2', filter_size=u.CNN.FILTER_SIZE),
    ])
    def test_predict(self, layer):
        N = (u.CNN.NUM_HEIGHT - u.CNN.FILTER_HEIGHT + 1) * \
            (u.CNN.NUM_WIDTH - u.CNN.FILTER_WIDTH + 1) * NH
        net = theanets.convolution.Regressor([
            NI, NH, layer, ('flat', N), u.NUM_OUTPUTS])
        assert net.predict(u.CNN.INPUTS).shape == (u.NUM_EXAMPLES, u.NUM_OUTPUTS)

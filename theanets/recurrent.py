# Copyright (c) 2012 Leif Johnson <leif@leifjohnson.net>
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

'''This file contains recurrent network structures.'''

import climate
import numpy as np
import numpy.random as rng
import theano
import theano.tensor as TT

#from theano.tensor.shared_randomstreams import RandomStreams
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

from . import feedforward as ff

logging = climate.get_logger(__name__)


class Network(ff.Network):
    '''A fully connected recurrent network with one input and one output layer.

    Parameters
    ----------
    layers : sequence of int
        A sequence of integers specifying the number of units at each layer. As
        an example, layers=(10, 20, 3) has one "input" layer with 10 units, one
        "hidden" layer with 20 units, and one "output" layer with 3 units. That
        is, inputs should be of length 10, and outputs will be of length 3.

    recurrent_layers : sequence of int, optional
        A sequence of integers specifying the indices of recurrent layers in the
        network. Non-recurrent network layers receive input only from the
        preceding layers for a given input, while recurrent layers also receive
        input from the output of the recurrent layer from the previous time
        step. Defaults to [len(layers) // 2 - 1], i.e., the "middle" layer of
        the network is the only recurrent layer.

    recurrent_sparsity : float in (0, 1), optional
        Ensure that the given fraction of recurrent model weights is initialized
        to zero. Defaults to 0, which makes all recurrent weights nonzero.

    hidden_activation : str, optional
        The name of an activation function to use on hidden network units.
        Defaults to 'sigmoid'.

    output_activation : str, optional
        The name of an activation function to use on output units. Defaults to
        'linear'.

    rng : theano.RandomStreams, optional
        Use a specific Theano random number generator. A new one will be created
        if this is None.

    input_noise : float, optional
        Standard deviation of desired noise to inject into input.

    hidden_noise : float, optional
        Standard deviation of desired noise to inject into hidden unit
        activation output.

    input_dropouts : float, optional
        Proportion of input units to randomly set to 0.

    hidden_dropouts : float, optional
        Proportion of hidden unit activations to randomly set to 0.

    recurrent_error_start : int, optional
        Compute error metrics starting at this time step. (Defaults to 3.)
    '''

    @property
    def error_start(self):
        return self.kwargs.get('recurrent_error_start', 3)

    def setup_vars(self):
        # the first dimension indexes time, the second indexes the elements of
        # each minibatch, and the third indexes the variables in a given frame.
        self.x = TT.tensor3('x')

    def setup_layers(self, **kwargs):
        '''
        '''
        count = 0

        noise = kwargs.get('input_noise', 0)
        dropout = kwargs.get('input_dropouts', 0)
        x = self._add_noise(self.x, noise, dropout)

        kw = dict(
            sparse=kwargs.get('recurrent_sparsity', 0),
            radius=kwargs.get('recurrent_radius', 0),
            noise=kwargs.get('hidden_noise', 0),
            dropout=kwargs.get('hidden_dropouts', 0),
        )
        layers = kwargs.get('layers')
        recurrent = set(kwargs.get('recurrent_layers', [len(layers) // 2 - 1]))
        for i, (nin, nout) in enumerate(zip(layers[:-1], layers[1:])):
            z = self.hiddens and self.hiddens[-1] or x
            add = self.add_feedforward_layer
            if i in recurrent:
                add = self.add_recurrent_layer
            count += add(z, nin, nout, label=i, **kw)

        self.hiddens.pop()
        self.y = self._output_func(self.preacts[-1])
        logging.info('%d total network parameters', count)

    def add_recurrent_layer(self, x, nin, nout, **kwargs):
        '''Add a new recurrent layer to the network.

        Parameters
        ----------
        input : theano variable
            The theano variable that represents the inputs to this layer.
        nin : int
            The number of input units to this layer.
        nout : out
            The number of output units from this layer.
        label : any, optional
            The name of this layer, used for logging and as the theano variable
            name suffix. Defaults to the index of this layer in the network.
        sparse : float in (0, 1), optional
            If given, create sparse connections in the recurrent weight matrix,
            such that this fraction of the weights is set to zero. By default,
            this parameter is 0, meaning all recurrent  weights are nonzero.
        radius : float, optional
            If given, rescale the initial weights to have this spectral radius.
            No scaling is performed by default.

        Returns
        -------
        count : int
            The number of learnable parameters in this layer.
        '''
        label = kwargs.get('label') or len(self.hiddens)

        b_h, _ = self.create_bias(nout, 'h_{}'.format(label))
        W_xh, _ = self.create_weights(nin, nout, 'xh_{}'.format(label))
        W_hh, _ = self.create_weights(nout, nout, 'hh_{}'.format(label), **kwargs)

        def fn(x_t, _, h_tm1, W_xh, W_hh, b_h):
            pre = TT.dot(x_t, W_xh) + TT.dot(h_tm1, W_hh) + b_h
            return pre, self._hidden_func(pre)

        batch_size = self.kwargs.get('batch_size', 64)
        (pre, hid), updates = theano.scan(
            name='rnn_{}'.format(label),
            fn=fn,
            sequences=[x],
            non_sequences=[W_xh, W_hh, b_h],
            outputs_info=[TT.zeros((batch_size, nout), dtype=ff.FLOAT),
                          TT.zeros((batch_size, nout), dtype=ff.FLOAT)])

        self.updates.update(updates)
        self.weights.extend([W_xh, W_hh])
        self.biases.append(b_h)
        self.preacts.append(pre)
        self.hiddens.append(hid)

        return nout * (1 + nin + nout)


class MRNN(Network):
    '''Define recurrent network layers using multiplicative dynamics.

    The formulation of MRNN implemented here uses a factored dynamics matrix as
    described in Sutskever, Martens & Hinton, ICML 2011, "Generating text with
    recurrent neural networks." This paper is available online at
    http://www.icml-2011.org/papers/524_icmlpaper.pdf.
    '''

    def add_recurrent_layer(self, x, nin, nout, **kwargs):
        '''Add a new recurrent layer to the network.

        Parameters
        ----------
        input : theano variable
            The theano variable that represents the inputs to this layer.
        nin : int
            The number of input units to this layer.
        nout : out
            The number of output units from this layer.
        label : any, optional
            The name of this layer, used for logging and as the theano variable
            name suffix. Defaults to the index of this layer in the network.
        factors : int, optional
            The number of factors to use in the hidden-to-hidden dynamics
            matrix. Defaults to the number of hidden units.

        Returns
        -------
        count : int
            The number of learnable parameters in this layer.
        '''
        label = kwargs.get('label') or len(self.hiddens)
        factors = kwargs.get('factors') or nout

        b_h, _ = self.create_bias(nout, 'h_{}'.format(label))

        W_xf, _ = self.create_weights(nin, factors, 'xf_{}'.format(label))
        W_hf, _ = self.create_weights(nout, factors, 'hf_{}'.format(label))
        W_fh, _ = self.create_weights(factors, nout, 'fh_{}'.format(label))
        W_xh, _ = self.create_weights(nin, nout, 'xh_{}'.format(label))

        def fn(x_t, _, h_tm1, W_xh, W_xf, W_hf, W_fh, b_h):
            f_t = TT.dot(TT.dot(h_tm1, W_hf) * TT.dot(x_t, W_xf), W_fh)
            pre = TT.dot(x_t, W_xh) + b_h + f_t
            return pre, self._hidden_func(pre)

        batch_size = self.kwargs.get('batch_size', 64)
        (pre, hid), self.updates = theano.scan(
            name='mrnn_{}'.format(label),
            fn=fn,
            sequences=[x],
            non_sequences=[W_xh, W_xf, W_hf, W_fh, b_h],
            outputs_info=[TT.zeros((batch_size, nout), dtype=ff.FLOAT),
                          TT.zeros((batch_size, nout), dtype=ff.FLOAT)])

        self.updates.update(updates)
        self.weights.extend([W_xh, W_xf, W_hf, W_fh])
        self.biases.append(b_h)
        self.preacts.append(pre)
        self.hiddens.append(hid)

        return nout * (1 + nin) + factors * (2 * nout + nin)


class LSTM(Network):
    '''
    '''

    def add_recurrent_layer(self, x, nin, nout, label=None):
        '''Add a new recurrent layer to the network.

        Parameters
        ----------
        input : theano variable
            The theano variable that represents the inputs to this layer.
        nin : int
            The number of input units to this layer.
        nout : out
            The number of output units from this layer.
        label : any, optional
            The name of this layer, used for logging and as the theano variable
            name suffix. Defaults to the index of this layer in the network.

        Returns
        -------
        count : int
            The number of learnable parameters in this layer.
        '''
        label = label or len(self.hiddens)

        b_i, _ = self.create_bias(nout, 'i_{}'.format(label))
        b_f, _ = self.create_bias(nout, 'f_{}'.format(label))
        b_o, _ = self.create_bias(nout, 'o_{}'.format(label))
        b_c, _ = self.create_bias(nout, 'c_{}'.format(label))

        # these weight matrices are always diagonal.
        W_ci, _ = self.create_bias(nout, 'ci_{}'.format(label))
        W_cf, _ = self.create_bias(nout, 'cf_{}'.format(label))
        W_co, _ = self.create_bias(nout, 'co_{}'.format(label))

        W_xi, _ = self.create_weights(nin, nout, 'xi_{}'.format(label))
        W_xf, _ = self.create_weights(nin, nout, 'xf_{}'.format(label))
        W_xo, _ = self.create_weights(nin, nout, 'xo_{}'.format(label))
        W_xc, _ = self.create_weights(nin, nout, 'xc_{}'.format(label))

        W_hi, _ = self.create_weights(nout, nout, 'hi_{}'.format(label))
        W_hf, _ = self.create_weights(nout, nout, 'hf_{}'.format(label))
        W_ho, _ = self.create_weights(nout, nout, 'ho_{}'.format(label))
        W_hc, _ = self.create_weights(nout, nout, 'hc_{}'.format(label))

        def fn(x_t, h_tm1, c_tm1, W_ci, W_cf, W_co, W_xi, W_xf, W_xo, W_xc, W_hi, W_hf, W_ho, W_hc, b_i, b_f, b_o, b_c):
            i_t = TT.nnet.sigmoid(TT.dot(x_t, W_xi) + TT.dot(h_tm1, W_hi) + c_tm1 * W_ci + b_i)
            f_t = TT.nnet.sigmoid(TT.dot(x_t, W_xf) + TT.dot(h_tm1, W_hf) + c_tm1 * W_cf + b_f)
            c_t = f_t * c_tm1 + i_t * TT.tanh(TT.dot(x_t, W_xc) + TT.dot(h_tm1, W_hc) + b_c)
            o_t = TT.nnet.sigmoid(TT.dot(x_t, W_xo) + TT.dot(h_tm1, W_ho) + c_t * W_co + b_o)
            h_t = o_t * TT.tanh(c_t)
            return h_t, c_t

        W = [W_ci, W_cf, W_co, W_xi, W_xf, W_xo, W_xc, W_hi, W_hf, W_ho, W_hc]
        B = [b_i, b_f, b_o, b_c]

        batch_size = self.kwargs.get('batch_size', 64)
        (hid, _), updates = theano.scan(
            name='f_{}'.format(label),
            fn=fn,
            sequences=[x],
            non_sequences=W + B,
            outputs_info=[TT.zeros((batch_size, nout), dtype=ff.FLOAT),
                          TT.zeros((batch_size, nout), dtype=ff.FLOAT)])

        self.updates.update(updates)
        self.weights.extend(W)
        self.biases.extend(B)
        self.preacts.append(hid)  # consider lstm output as preactivation
        self.hiddens.append(hid)

        return nout * (7 + 4 * nout + 4 * nin)


class Autoencoder(Network):
    '''An autoencoder attempts to reproduce its input.'''

    @property
    def cost(self):
        err = self.y - self.x
        return TT.mean((err * err).sum(axis=2)[self.error_start:])


class Predictor(Autoencoder):
    '''A predictor network attempts to predict its next time step.
    '''

    @property
    def cost(self):
        # we want the network to predict the next time step. y is the prediction
        # (output of the network), so we want y[0] to match x[1], y[1] to match
        # x[2], and so forth.
        err = self.x[1:] - self.generate_prediction(self.y)[:-1]
        return TT.mean((err * err).sum(axis=2)[self.error_start:])

    def generate_prediction(self, y):
        '''Given outputs from each time step, map them to subsequent inputs.

        This defaults to the identity transform, i.e., the output from one time
        step is treated as the input to the next time step with no
        transformation. Override this method in a subclass to provide, e.g.,
        predictions based on random samples, lookups in a dictionary, etc.

        Parameters
        ----------
        y : theano variable
            A symbolic variable representing the "raw" output of the recurrent
            predictor.

        Returns
        -------
        y : theano variable
            A symbolic variable representing the inputs for the next time step.
        '''
        return y


class Regressor(Network):
    '''A regressor attempts to produce a target output.'''

    def setup_vars(self):
        self.x = TT.tensor3('x')
        self.k = TT.tensor3('k')

    @property
    def inputs(self):
        return [self.x, self.k]

    @property
    def cost(self):
        err = self.y - self.k
        return TT.mean((err * err).sum(axis=2)[self.error_start:])


class Classifier(Network):
    '''A classifier attempts to match a 1-hot target output.'''

    def __init__(self, **kwargs):
        kwargs['output_activation'] = 'softmax'
        super(Classifier, self).__init__(**kwargs)

    def setup_vars(self):
        super(Classifier, self).setup_vars()

        # for a classifier, k specifies the correct labels for a given input.
        self.k = TT.ivector('k')

    @property
    def inputs(self):
        return [self.x, self.k]

    @property
    def cost(self):
        return -TT.mean(TT.log(self.y)[TT.arange(self.k.shape[0]), self.k])

    @property
    def accuracy(self):
        '''Compute the percent correct classifications.'''
        return 100 * TT.mean(TT.eq(TT.argmax(self.y, axis=1), self.k))

    @property
    def monitors(self):
        yield 'acc', self.accuracy
        for i, h in enumerate(self.hiddens):
            yield 'h{}<0.1'.format(i+1), 100 * (abs(h) < 0.1).mean()
            yield 'h{}<0.9'.format(i+1), 100 * (abs(h) < 0.9).mean()

    def classify(self, x):
        return self.predict(x).argmax(axis=1)

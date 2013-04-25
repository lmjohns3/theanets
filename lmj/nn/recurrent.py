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

import numpy as np
import numpy.random as rng
import theano
import theano.tensor as TT
from theano.tensor.shared_randomstreams import RandomStreams

from . import feedforward as ff
from . import log

logging = log.get_logger(__name__)


class Network(ff.Network):
    '''A fully connected recurrent network with one input and one output layer.
    '''

    def __init__(self, layers, activation, rng=None, input_noise=0,
                 hidden_noise=0, input_dropouts=0, hidden_dropouts=0, **kwargs):
        '''Initialize the weights and computation graph for a recurrent network.

        layers: A sequence of three integers specifying the number of units in
          the input, hidden, and output layers, respectively.
        activation: A callable that takes one argument (a matrix) and returns
          another matrix. This is the activation function that each hidden unit
          in the network uses.
        rng: Use a specific Theano random number generator. A new one will be
          created if this is None.
        input_noise: Standard deviation of desired noise to inject into input.
        hidden_noise: Standard deviation of desired noise to inject into
          hidden unit activation output.
        input_dropouts: Proportion of input units to randomly set to 0.
        hidden_dropouts: Proportion of hidden unit activations to randomly set
          to 0.

        Recognized keyword arguments:

        damping: This parameter (a float in [0, 1]) governs the proportion of
          past state information retained by the hidden neurons.
        '''
        nin, nhid, nout = layers

        # in this module, x refers to a network's input, and y to its output.
        self.x = TT.matrix('x')
        self.x.tag.test_value = ff.randn(ff.DEBUG_BATCH_SIZE, nin)

        arr = ff.randn(nin, nhid) / np.sqrt(nin + nhid)
        W_in = theano.shared(arr.astype(ff.FLOAT), name='W_in')
        logging.info('inputs: %d x %d', nin, nhid)

        arr = ff.randn(nhid, nhid) / np.sqrt(nhid + nhid) / 10.
        W_pool = theano.shared(arr.astype(ff.FLOAT), name='W_pool')
        b_pool = theano.shared(np.zeros((nhid, ), ff.FLOAT), name='b_pool')
        logging.info('hidden: %d x %d', nhid, nhid)

        arr = ff.randn(nhid, nout) / np.sqrt(nhid + nout)
        W_out = theano.shared(arr.astype(ff.FLOAT), name='W_out')
        b_out = theano.shared(np.zeros((nout, ), ff.FLOAT), name='b_out')
        logging.info('outputs: %d x %d', nhid, nout)

        logging.info('%d total network parameters',
                     nhid * (nin + nhid + nout + 1) + nout)

        rng = rng or RandomStreams()

        damping = kwargs.get('damping')
        if damping is None:
            damping = 0.01

        def step(x_t, h_tm1):
            if input_noise > 0:
                x_t += rng.normal(size=x_t.shape, std=input_noise)
            if input_dropouts > 0:
                x_t *= rng.uniform(low=0, high=1, ndim=2) > input_dropouts
            h_t = activation(TT.dot(x_t, W_in) + TT.dot(h_tm1, W_pool) + b_pool)
            h_t.tag.test_value = ff.randn(ff.DEBUG_BATCH_SIZE, nhid)
            if hidden_noise > 0:
                h_t += rng.normal(size=h_t.shape, std=hidden_noise)
            if hidden_dropouts > 0:
                h_t *= rng.uniform(low=0, high=1, ndim=2) > hidden_dropouts
            h_t = (1 - damping) * h_t + damping * h_tm1
            return [h_t, TT.dot(h_t, W_out) + b_out]

        h_0 = theano.shared(np.zeros((nhid, ), ff.FLOAT), 'h_0')
        (h, self.y), self.updates = theano.scan(
            fn=step, sequences=self.x, outputs_info=[h_0, {}])

        self.hiddens = [h]
        self.weights = [W_in, W_pool, W_out]
        self.biases = [b_pool, b_out]

        # compute a complete pass over an input sequence.
        self.forward = theano.function(
            [self.x], self.hiddens + [self.y], updates=self.updates)


class Autoencoder(Network):
    '''An autoencoder attempts to reproduce its input.'''

    @property
    def cost(self):
        err = self.y - self.x
        return TT.mean((err * err).sum(axis=1))


class Regressor(Network):
    '''A regressor attempts to produce a target output.'''

    def __init__(self, *args, **kwargs):
        super(Regressor, self).__init__(*args, **kwargs)
        self.k = TT.matrix('k')

    @property
    def inputs(self):
        return [self.x, self.k]

    @property
    def cost(self):
        err = self.k - self.y
        return TT.mean((err * err).sum(axis=1))

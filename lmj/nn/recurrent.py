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

import lmj.cli
import numpy as np
import numpy.random as rng
import theano
import theano.tensor as TT
from theano.tensor.shared_randomstreams import RandomStreams

from . import feedforward as ff

logging = lmj.cli.get_logger(__name__)


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
        layers = list(layers)
        num_out = layers.pop()
        num_pool = layers.pop()

        # in this module, x refers to a network's input, and y to its output.
        self.x = TT.matrix('x')
        self.x.tag.test_value = ff.randn(ff.DEBUG_BATCH_SIZE, layers[0])

        parameter_count = 0

        W_in = []
        b_in = []
        for i, (a, b) in enumerate(zip(layers, layers[1:] + [num_pool])):
            arr = ff.randn(a, b) / np.sqrt(a + b)
            W_in.append(theano.shared(arr.astype(ff.FLOAT), name='weights_%d' % i))
            b_in.append(theano.shared(np.zeros((b, ), ff.FLOAT), name='bias_%d' % i))
            logging.info('encoding weights for layer %d: %d x %d', i + 1, a, b)
            parameter_count += (1 + a) * b

        # discard bias for last input layer (we create it explicitly below, as
        # the bias for the recurrent pool)
        b_in.pop()
        parameter_count -= num_pool

        arr = ff.randn(num_pool, num_pool) / np.sqrt(num_pool + num_pool) / 10.
        W_pool = theano.shared(arr.astype(ff.FLOAT), name='weights_pool')
        b_pool = theano.shared(np.zeros((num_pool, ), ff.FLOAT), name='bias_pool')
        logging.info('recurrent weights at layer %d: %d x %d',
                     len(W_in), num_pool, num_pool)
        parameter_count += (1 + num_pool) * num_pool

        arr = ff.randn(num_pool, num_out) / np.sqrt(num_pool + num_out)
        W_out = theano.shared(arr.astype(ff.FLOAT), name='weights_out')
        b_out = theano.shared(np.zeros((num_out, ), ff.FLOAT), name='bias_out')
        logging.info('output weights at layer %d: %d x %d',
                     len(W_in) + 1, num_pool, num_out)
        parameter_count += (1 + num_pool) * num_out

        logging.info('%d total network parameters', parameter_count)

        rng = rng or RandomStreams()

        damping = kwargs.get('damping')
        if damping is None:
            damping = 0.01

        def step(x_t, h_tm1):
            z = x_t
            if input_noise > 0:
                z += rng.normal(size=z.shape, std=input_noise)
            if input_dropouts > 0:
                z *= rng.uniform(low=0, high=1, ndim=2) > input_dropouts
            for i, (W, b) in enumerate(zip(W_in, b_in)):
                z = activation(TT.dot(z, W) + b)
                if hidden_noise > 0:
                    z += rng.normal(size=z.shape, std=hidden_noise)
                if hidden_dropouts > 0:
                    z *= rng.uniform(low=0, high=1, ndim=2) > hidden_dropouts
            h_t = activation(TT.dot(z, W_in[-1]) + TT.dot(h_tm1, W_pool) + b_pool)
            h_t = (1 - damping) * h_t + damping * h_tm1
            return [h_t, TT.dot(h_t, W_out) + b_out]

        h_0 = theano.shared(np.zeros(num_pool).astype(ff.FLOAT), name='h_0')
        (h, self.y), self.updates = theano.scan(
            fn=step, sequences=self.x, outputs_info=[h_0, {}])

        self.hiddens = [h]
        self.weights = W_in + [W_pool, W_out]
        self.biases = b_in + [b_pool, b_out]

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

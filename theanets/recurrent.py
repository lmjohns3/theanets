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

from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

from . import feedforward as ff

logging = climate.get_logger(__name__)


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

        pool_damping: This parameter (a float in [0, 1]) governs the proportion
          of past state information retained by the recurrent pool neurons.
        pool_noise: Add gaussian noise to recurrent pool neurons with this
          variance.
        pool_dropouts: Randomly set the state of this fraction (a float in
          [0, 1]) of recurrent pool neurons to zero.
        pool_error_start: Compute error metrics starting at this time step.
        '''
        self.rng = rng or RandomStreams()
        pool_noise = kwargs.get('pool_noise', 0.)
        pool_dropouts = kwargs.get('pool_dropouts', 0.)
        pool_damping = kwargs.get('pool_damping', 0.)
        self.error_start = kwargs.get('pool_error_start', 3)

        # in this module, x refers to a network's input, and y to its output.
        self.x = TT.matrix('x')

        parameter_count = 0
        layers = list(layers)
        num_out = layers.pop()
        num_pool = layers.pop()

        W_in = []
        b_in = []
        for i, (a, b) in enumerate(zip(layers, layers[1:] + [num_pool])):
            W, b, params = self._weights_and_bias(a, b, i)
            parameter_count += params
            W_in.append(W)
            b_in.append(b)

        # discard bias for last input layer (we create it explicitly below, as
        # it overlaps with the bias for the recurrent pool)
        b_in.pop()
        parameter_count -= num_pool

        W_pool, b_pool, params = self._weights_and_bias(num_pool, num_pool, 'pool')
        parameter_count += params

        W_out, b_out, params = self._weights_and_bias(num_pool, num_out, 'out')
        parameter_count += params

        logging.info('%d total network parameters', parameter_count)

        def step(x_t, h_tm1):
            z = self._noise_and_dropout(x_t, input_noise, input_dropouts)
            encs = []
            for W, b in zip(W_in, b_in):
                encs.append(self._noise_and_dropout(
                    activation(TT.dot(z, W) + b), hidden_noise, hidden_dropouts))
                z = encs[-1]
            h = activation(TT.dot(z, W_in[-1]) + TT.dot(h_tm1, W_pool) + b_pool)
            h_t = self._noise_and_dropout(
                (1 - pool_damping) * h + pool_damping * h_tm1,
                pool_noise, pool_dropouts)
            return encs + [h_t, TT.dot(h_t, W_out) + b_out]

        h_0 = TT.zeros((num_pool, ), dtype=ff.FLOAT)
        outputs, self.updates = theano.scan(
            fn=step, sequences=self.x,
            outputs_info=[None for _ in b_in] + [h_0, None])

        self.y = outputs.pop()
        self.hiddens = outputs
        self.weights = W_in + [W_pool, W_out]
        self.biases = b_in + [b_pool, b_out]

        # compute a complete pass over an input sequence.
        self.feed_forward = theano.function(
            [self.x], self.hiddens + [self.y], updates=self.updates)


class Autoencoder(Network):
    '''An autoencoder attempts to reproduce its input.'''

    @property
    def cost(self):
        err = self.y - self.x
        return TT.mean((err * err).sum(axis=1)[self.error_start:])


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
        err = self.y - self.k
        return TT.mean((err * err).sum(axis=1)[self.error_start:])

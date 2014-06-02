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
        A sequence of three integers specifying the number of units in the
        input, hidden, and output layers, respectively.

    activation : callable(numeric) -> numeric
        A callable that takes one argument (a matrix) and returns another
        matrix. This is the activation function that each hidden unit in the
        network uses.

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

    pool_noise : float, optional
        Add gaussian noise to recurrent pool neurons with this variance.

    pool_dropouts : float in [0, 1], optional
        Randomly set the state of this fraction of recurrent pool neurons to
        zero.

    pool_error_start : int, optional
        Compute error metrics starting at this time step. (Defaults to 3.)
    '''

    def __init__(self, layers, activation, rng=None, **kwargs):
        self.layers = tuple(layers)
        self.activation = activation
        self.hiddens = []
        self.weights = []
        self.biases = []

        self.rng = kwargs.get('rng') or RandomStreams()
        self.error_start = kwargs.get('pool_error_start', 3)

        # the first dimension indexes time, the second indexes the elements of
        # each minibatch, and the third indexes the variables in a given frame.
        self.x = TT.tensor3('x')

        activation = self._build_activation(activation)
        if hasattr(activation, '__theanets_name__'):
            logging.info('hidden activation: %s', activation.__theanets_name__)

        sizes = list(layers)
        num_out = sizes.pop()
        num_pool = sizes.pop()

        z, parameter_count = self._create_forward_map(sizes, activation, **kwargs)

        # set up a recurrent computation graph to pass hidden states in time.
        W_in, _, count = self._create_layer(sizes[-1], num_pool, 'in')
        parameter_count += count - num_pool
        W_pool, b_pool, count = self._create_layer(num_pool, num_pool, 'pool')
        parameter_count += count
        W_out, b_out, count = self._create_layer(num_pool, num_out, 'out')
        parameter_count += count
        logging.info('%d total network parameters', parameter_count)

        def recurrence(z_t, h_tm1):
            return activation(TT.dot(z_t, W_in) + TT.dot(h_tm1, W_pool) + b_pool)

        batch_size = kwargs.get('batch_size', 64)
        h_0 = TT.zeros((batch_size, num_pool), dtype=ff.FLOAT)
        h, self.updates = theano.scan(
            fn=recurrence, sequences=z, outputs_info=[h_0])

        # map hidden states to outputs using a linear transform.
        self.y = TT.dot(h, W_out) + b_out

        self.hiddens.append(h)
        self.weights.extend([W_in, W_pool, W_out])
        self.biases.extend([b_pool, b_out])


class Autoencoder(Network):
    '''An autoencoder attempts to reproduce its input.'''

    @property
    def cost(self):
        err = self.y - self.x
        return TT.mean((err * err).sum(axis=2)[self.error_start:])


class Predictor(Autoencoder):
    '''A predictor network attempts to predict its next time step.'''

    @property
    def cost(self):
        # we want the network to predict the next time step. y is the
        # prediction, so we want y[0] to match x[1], y[1] to match x[2], and so
        # forth.
        err = self.x[1:] - self.y[:-1]
        return TT.mean((err * err).sum(axis=2)[self.error_start:])


class Regressor(Network):
    '''A regressor attempts to produce a target output.'''

    def __init__(self, *args, **kwargs):
        super(Regressor, self).__init__(*args, **kwargs)
        self.k = TT.tensor3('k')

    @property
    def inputs(self):
        return [self.x, self.k]

    @property
    def cost(self):
        err = self.y - self.k
        return TT.mean((err * err).sum(axis=2)[self.error_start:])

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
        self.rng = rng or RandomStreams()
        self.error_start = kwargs.get('pool_error_start', 3)

        # the first dimension indexes time, the second indexes the elements of
        # each minibatch, and the third indexes the variables in a given frame.
        self.x = TT.tensor3('x')

        parameter_count = 0
        layers = list(layers)
        num_out = layers.pop()
        num_pool = layers.pop()

        W_in = []
        b_in = []
        for i, (a, b) in enumerate(zip(layers, layers[1:] + [num_pool])):
            W, b, count = self._create_layer(a, b, i)
            parameter_count += count
            W_in.append(W)
            b_in.append(b)

        # we give a special name to the bias for the recurrent pool units.
        b_pool = b_in.pop()

        W_pool, _, count = self._create_layer(num_pool, num_pool, 'pool')
        parameter_count += count - num_pool

        W_out, b_out, count = self._create_layer(num_pool, num_out, 'out')
        parameter_count += count

        logging.info('%d total network parameters', parameter_count)

        def recurrence(x_t, h_tm1):
            z = [self._add_noise(x_t,
                                 kwargs.get('input_noise', 0.),
                                 kwargs.get('input_dropouts', 0.))]
            for W, b in zip(W_in, b_in):
                z.append(self._add_noise(
                    activation(TT.dot(z[-1], W) + b),
                    kwargs.get('hidden_noise', 0.),
                    kwargs.get('hidden_dropouts', 0.)))
            z.append(self._add_noise(activation(
                TT.dot(z[-1], W_in[-1]) + TT.dot(h_tm1, W_pool) + b_pool),
                kwargs.get('pool_noise', 0.),
                kwargs.get('pool_dropouts', 0.)))
            z.append(TT.dot(z[-1], W_out) + b_out)
            return z[1:]

        batch_size = kwargs.get('batch_size', 64)
        h_0 = TT.zeros((batch_size, num_pool), dtype=ff.FLOAT)
        outputs, self.updates = theano.scan(
            fn=recurrence,
            sequences=self.x,
            outputs_info=[None for _ in b_in] + [h_0, None])

        self.y = outputs.pop()
        self.hiddens = outputs
        self.weights = W_in + [W_pool, W_out]
        self.biases = b_in + [b_pool, b_out]


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
        # prediction, so we want y[0] to match x[1], y[1] to match x[2], at so forth.
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

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
        step. The index values in this sequence must be in (0, len(layers)) --
        that is, the input and output of a network cannot be recurrent. Defaults
        to [len(layers) - 1] -- the penultimate layer of the network is the only
        recurrent layer.

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

    def setup_vars(self):
        # the first dimension indexes time, the second indexes the elements of
        # each minibatch, and the third indexes the variables in a given frame.
        self.x = TT.tensor3('x')

    def setup_encoder(self, **kwargs):
        self.error_start = kwargs.get('recurrent_error_start', 3)

        sizes = self.check_layer_sizes()

        x, forward_count = super(Network, self).setup_encoder(**kwargs)
        f, recurrent_count = self.setup_recurrence(sizes[-1])

        self.hiddens.pop()
        batch_size = kwargs.get('batch_size', 64)
        h_0 = TT.zeros((batch_size, sizes[-1]), dtype=ff.FLOAT)
        z = self.hiddens[-1] if self.hiddens else x
        h, up = theano.scan(fn=f, sequences=z, outputs_info=[h_0])
        self.updates.update(up)
        self.hiddens.append(h)

        return h, forward_count + recurrent_count

    def setup_recurrence(self, size):
        # once we've set up the encoding layers, we add a recurrent connection
        # on the topmost layer. this entails creating a new weight matrix
        # W_pool, but we reuse the existing bias values.
        W_in = self.weights[-1]
        b_pool = 0 if self.tied_weights else self.biases[-1]
        W_pool, _, count = self.create_layer(size, size, 'pool')
        count -= size

        def recurrence(z_t, h_tm1):
            return self._hidden_func(TT.dot(z_t, W_in) + TT.dot(h_tm1, W_pool) + b_pool)

        self.weights.append(W_pool)

        return recurrence, count

    def lstm_recurrence(self, input_size, cell_size):
        count = 0

        W_xi, b_i, n = self.create_layer(input_size, cell_size, 'xi')
        count += n
        W_hi, W_ci, n = self.create_layer(cell_size, cell_size, 'hi')
        W_ci.name = 'W_ci'
        count += n

        W_xf, b_f, n = self.create_layer(input_size, cell_size, 'xf')
        count += n
        W_hf, W_cf, n = self.create_layer(cell_size, cell_size, 'hf')
        W_cf.name = 'W_cf'
        count += n

        W_xo, b_o, n = self.create_layer(input_size, cell_size, 'xo')
        count += n
        W_ho, W_co, n = self.create_layer(cell_size, cell_size, 'ho')
        W_co.name = 'W_co'
        count += n

        W_xc, b_c, n = self.create_layer(input_size, cell_size, 'xc')
        count += n
        W_hc, _, n = self.create_layer(cell_size, cell_size, 'hc')
        count += n - cell_size

        def recurrence(x_t, h_tm1, c_tm1):
            i_t = TT.nnet.sigmoid(TT.dot(x_t, W_xi) +
                                  TT.dot(h_tm1, W_hi) +
                                  TT.dot(c_tm1, TT.diag(W_ci)) + b_i)
            f_t = TT.nnet.sigmoid(TT.dot(x_t, W_xf) +
                                  TT.dot(h_tm1, W_hf) +
                                  TT.dot(c_tm1, TT.diag(W_cf)) + b_f)
            c_t = f_t * c_tm1 + i_t * TT.tanh(TT.dot(x_t, W_xc) +
                                              TT.dot(h_tm1, W_hc) + b_c)
            o_t = TT.nnet.sigmoid(TT.dot(x_t, W_xo) +
                                  TT.dot(h_tm1, W_ho) +
                                  TT.dot(c_t, TT.diag(W_co)) + b_o)
            h_t = o_t * TT.tanh(c_t)
            return h_t, c_t

        return recurrence, count


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
        # we want the network to predict the next time step. y is the prediction
        # (output of the network), so we want y[0] to match x[1], y[1] to match
        # x[2], and so forth.
        err = self.x[1:] - self.y[:-1]
        return TT.mean((err * err).sum(axis=2)[self.error_start:])


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

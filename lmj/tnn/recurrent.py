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

import numpy
import numpy.random as rng
import theano
import theano.tensor as T

import .feedforward


class Network(feedforward.Network):
    '''A fully connected recurrent network with inputs and outputs.'''

    def __init__(self, layers, nonlinearity=TT.nnet.sigmoid, gain=1.2, damping=0.01):
        self.x = T.matrix('x')
        self.state = theano.shared(
            numpy.zeros(size, dtype=feedforward.FLOAT), name='state')

        i, h, o = layers
        arr = rng.normal(size=(i, h)) / numpy.sqrt(i + h)
        W_in = theano.shared(arr.astype(feedforward.FLOAT), name='W_in')

        arr = gain * rng.normal(size=(h, h)) / numpy.sqrt(h + h)
        W_pool = theano.shared(arr.astype(feedforward.FLOAT), name='W_pool')

        arr = rng.normal(size=(h, o)) / numpy.sqrt(h + o)
        W_out = theano.shared(arr.astype(feedforward.FLOAT), name='W_out')
        b_out = theano.shared(numpy.zeros((o, ), feedforward.FLOAT), name='b_out')

        self.hiddens = [self.state]
        self.weights = [W_in, W_pool, W_out]
        self.biases = [b_out]

        logging.info('%d total network parameters', h * (i + h + o) + o)

        z = nonlinearity(T.dot(self.x, W_in) + T.dot(self.state, W_pool) + b_pool)
        self.next_state = damping * self.state + (1 - damping) * z
        self.y = T.dot(self.next_state, W_out) + b_out

        self.f = theano.function(*self.args, updates={self.state: self.next_state})

    @property
    def inputs(self):
        return [self.x]

    @property
    def args(self):
        return [self.x], [self.y]

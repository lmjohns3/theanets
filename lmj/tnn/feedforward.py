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

'''This module contains a number of classes for modeling neural nets in Theano.
'''

import cPickle
import gzip
import logging
import numpy
import numpy.random as rng
import theano
import theano.tensor as TT


FLOAT = theano.config.floatX


class Network(object):
    '''The network class is a fairly basic fully-connected feedforward net.
    '''

    def __init__(self, layers, activation, decode=1):
        '''Create a new feedforward network of a specific topology.

        layers: A sequence of integers specifying the number of units at each
          layer. As an example, layers=(10, 20, 3) has one "input" layer with 10
          units, one "hidden" layer with 20 units, and one "output" layer with 3
          units. That is, inputs should be of length 10, and outputs will be of
          length 3.
        activation: A callable that takes one argument (a matrix) and returns
          another matrix. This is the activation function that each hidden unit
          in the network uses.
        decode: Any of the hidden layers can be tapped at the output. Just
          specify a value greater than 1 to tap the last N hidden layers.
        '''
        # in this module, x refers to a network's input, and y to its output.
        self.x = TT.matrix('x')

        self.hiddens = []
        self.weights = []
        self.biases = []

        count = 0
        for i, (a, b) in enumerate(zip(layers[:-2], layers[1:-1])):
            count += (1 + a) * b
            logging.info('encoding weights for layer %d: %s x %s', i + 1, a, b)
            arr = rng.normal(size=(a, b)) / numpy.sqrt(a + b)
            Wi = theano.shared(arr.astype(FLOAT), name='W_%d' % i)
            bi = theano.shared(numpy.zeros((b, ), FLOAT), name='b_%d' % i)
            z = activation(TT.dot(self.hiddens[-1] if i else self.x, Wi) + bi)
            self.hiddens.append(z)
            self.weights.append(Wi)
            self.biases.append(bi)

        k = layers[-1]
        decoders = []
        for i, W in enumerate(reversed(self.weights[-decode:])):
            i = len(self.weights) - i
            b = W.get_value(borrow=True).shape[1]
            count += b * k
            logging.info('decoding weights from layer %d: %s x %s', i, b, k)
            arr = rng.normal(size=(b, k)) / numpy.sqrt(b + k)
            decoders.append(theano.shared(arr.astype(FLOAT), name='decode_%d' % i))
        count += k
        bias = theano.shared(numpy.zeros((k, ), FLOAT), name='b_out')
        self.weights.extend(decoders)
        self.biases.append(bias)

        logging.info('%d total network parameters', count)

        self.y = sum(TT.dot(*z) for z in zip(self.hiddens[::-1], decoders[::-1])) + bias
        self.forward = theano.function(*self.args)
        self.encode = theano.function(self.inputs, self.hiddens)

    @property
    def inputs(self):
        return [self.x]

    @property
    def args(self):
        return [self.x], [self.y]

    @property
    def monitors(self):
        return self.sparsities

    @property
    def params(self):
        return self.weights + self.biases

    @property
    def sparsities(self):
        return [TT.eq(h, 0).mean() for h in self.hiddens]

    @property
    def covariances(self):
        return [TT.dot(h.T, h) for h in self.hiddens]

    def __call__(self, *inputs):
        '''Compute a forward pass of the given inputs, returning the net output.
        '''
        return self.forward(*inputs)

    def save(self, filename):
        '''Save the parameters of this network to disk.'''
        opener = gzip.open if filename.lower().endswith('.gz') else open
        handle = opener(filename, 'wb')
        cPickle.dump([param.get_value().copy() for param in self.params], handle, -1)
        handle.close()

    def load(self, filename):
        '''Load the parameters for this network from disk.'''
        opener = gzip.open if filename.lower().endswith('.gz') else open
        handle = opener(filename, 'rb')
        for param, s in zip(self.params, cPickle.load(handle)):
            param.set_value(s)
        handle.close()
        logging.info('%s: loaded model parameters', filename)

    def J(self, weight_l1=None, weight_l2=None, activity_l1=None, **unused):
        '''Return a cost function for this network.'''
        cost = self.cost
        if weight_l1 > 0:
            cost += weight_l1 * sum(abs(i).mean(axis=0).sum() for i in self.weights)
        if weight_l2 > 0:
            cost += weight_l2 * sum(abs(i * i).mean(axis=0).sum() for i in self.weights)
        if activity_l1 > 0:
            cost += activity_l1 * sum(abs(h).mean(axis=0).sum() for h in self.hiddens)
        return cost


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


class Classifier(Network):
    '''A classifier attempts to match a 1-hot target output.'''

    def __init__(self, *args, **kwargs):
        super(Classifier, self).__init__(*args, **kwargs)
        self.y = self.softmax(self.y)
        self.k = TT.ivector('k')

    @staticmethod
    def softmax(x):
        # TT.nnet.softmax doesn't work with the HF trainer.
        z = TT.exp(x - x.max(axis=1)[:, None])
        return z / z.sum(axis=1)[:, None]

    @property
    def inputs(self):
        return [self.x, self.k]

    @property
    def prediction(self):
        return TT.argmax(self.y, axis=1)

    @property
    def cost(self):
        return -TT.mean(TT.log(self.y)[TT.arange(self.k.shape[0]), self.k])

    @property
    def incorrect(self):
        return TT.mean(TT.neq(self.prediction, self.k))

    @property
    def monitors(self):
        return [self.incorrect] + self.sparsities

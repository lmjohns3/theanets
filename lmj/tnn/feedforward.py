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

import pickle
import gzip
import logging
import numpy
import theano
import theano.tensor as TT
from theano.tensor.shared_randomstreams import RandomStreams

FLOAT = theano.config.floatX
DEBUG_BATCH_SIZE = 11


class Network(object):
    '''The network class is a fully-connected feedforward net.

    It can add random noise to the inputs, or to the hidden unit activations.
    Adding noise to these quantities can be seen as a form of regularization.

    This class also permits "decoding" (computing final network output) from
    more than just the final hidden layer of units ; however, decoding must
    always include the final k hidden layers in the network.
    '''

    def __init__(self, layers, activation, decode=1,
                 rng=None, input_noise=0, hidden_noise=0,
                 input_dropouts=0, hidden_dropouts=0):
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
        rng: Use a specific Theano random number generator. A new one will be
          created if this is None.
        input_noise: Standard deviation of desired noise to inject into input.
        hidden_noise: Standard deviation of desired noise to inject into
          hidden unit activation output.
        input_dropouts: Proportion of input units to randomly set to 0.
        hidden_dropouts: Proportion of hidden unit activations to randomly set
          to 0.
        '''
        # in this module, x refers to a network's input, and y to its output.
        self.x = TT.matrix('x')

        self.hiddens = []
        self.weights = []
        self.biases = []
        self.gains = []

        rng = rng or RandomStreams()
        randn = numpy.random.randn

        count = 0
        for i, (a, b) in enumerate(zip(layers[:-2], layers[1:-1])):
            if i == 0:
                # for shape debugging
                self.x.tag.test_value = randn(DEBUG_BATCH_SIZE, a)

            count += (1 + a) * b
            logging.info('encoding weights for layer %d: %s x %s', i + 1, a, b)

            arr = numpy.random.normal(size=(a, b)) / numpy.sqrt(a + b)
            Wi = theano.shared(arr.astype(FLOAT), name='weights_%d' % i)
            Wi.tag.test_value = randn(a, b)
            bi = theano.shared(numpy.zeros((b, ), FLOAT), name='bias_%d' % i)
            bi.tag.test_value = randn(b)
            gi = theano.shared(numpy.ones((b, ), FLOAT), name='gain_%d' % i)
            gi.tag.test_value = randn(b)

            x = self.x if i == 0 else self.hiddens[-1]
            if input_noise > 0 and i == 0:
                x += rng.normal(size=x.shape, std=input_noise)
            if input_dropouts > 0 and i == 0:
                x *= rng.uniform(low=0, high=1, size=x.shape) > input_dropouts

            z = activation(gi * (TT.dot(x, Wi) + bi))
            z.tag.test_value = randn(DEBUG_BATCH_SIZE, b)

            if hidden_noise > 0:
                z += rng.normal(size=z.shape, std=hidden_noise)
            if hidden_dropouts > 0:
                z *= rng.uniform(low=0, high=1, size=z.shape) > hidden_dropouts

            self.hiddens.append(z)
            self.weights.append(Wi)
            self.biases.append(bi)
            self.gains.append(gi)

        n = layers[-1]
        w = len(self.weights)
        terms = []
        for i in range(w - 1, w - 1 - decode, -1):
            W = self.weights[i]
            b = self.biases[i].get_value(borrow=True).shape[0]
            logging.info('decoding weights from layer %d: %s x %s', i + 1, b, n)
            count += b * n
            arr = numpy.random.normal(size=(b, n)) / numpy.sqrt(b + n)
            Di = theano.shared(arr.astype(FLOAT), name='decode_%d' % i)
            Di.tag.test_value = randn(b, n)
            terms.append(TT.dot(self.hiddens[i], Di))
            self.weights.append(Di)

        bias = theano.shared(numpy.zeros((n, ), FLOAT), name='bias_out')
        count += n

        logging.info('%d total network parameters', count)

        self.y = sum(terms) + bias
        self.f = theano.function([self.x], self.hiddens + [self.y])

    @property
    def inputs(self):
        return [self.x]

    @property
    def monitors(self):
        return self.sparsities

    @property
    def sparsities(self):
        return [TT.eq(h, 0).mean() for h in self.hiddens]

    def params(self, **kwargs):
        params = self.weights
        if not kwargs.get('nolearn_biases'):
            params.extend(self.biases)
        if kwargs.get('learn_gains'):
            params.extend(self.gains)
        return params

    def __call__(self, x):
        '''Compute a forward pass of the given inputs, returning the net output.
        '''
        return self.f(x)

    def save(self, filename):
        '''Save the parameters of this network to disk.'''
        opener = gzip.open if filename.lower().endswith('.gz') else open
        handle = opener(filename, 'wb')
        pickle.dump(
            dict(weights=[p.get_value().copy() for p in self.weights],
                 biases=[p.get_value().copy() for p in self.biases],
                 gains=[p.get_value().copy() for p in self.gains],
                 ), handle, -1)
        handle.close()
        logging.info('%s: saved model parameters', filename)

    def load(self, filename):
        '''Load the parameters for this network from disk.'''
        opener = gzip.open if filename.lower().endswith('.gz') else open
        handle = opener(filename, 'rb')
        params = pickle.load(handle)
        for target, source in zip(self.weights, params['weights']):
            target.set_value(source)
        for target, source in zip(self.biases, params['biases']):
            target.set_value(source)
        for target, source in zip(self.biases, params['gains']):
            target.set_value(source)
        handle.close()
        logging.info('%s: loaded model parameters', filename)

    def J(self, weight_l1=0, weight_l2=0, hidden_l1=0, hidden_l2=0, **unused):
        '''Return a cost function for this network.'''
        cost = self.cost
        if weight_l1 > 0:
            cost += weight_l1 * sum(abs(w).sum() for w in self.weights)
        if weight_l2 > 0:
            cost += weight_l2 * sum((w * w).sum() for w in self.weights)
        if hidden_l1 > 0:
            cost += hidden_l1 * sum(abs(h).mean(axis=0).sum() for h in self.hiddens)
        if hidden_l2 > 0:
            cost += hidden_l2 * sum((h * h).mean(axis=0).sum() for h in self.hiddens)
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
        self.k = TT.matrix('k')
        super(Regressor, self).__init__(*args, **kwargs)
        # for shape debugging
        w = self.weights[len(self.biases)]
        self.k.tag.test_value = numpy.random.randn(
            DEBUG_BATCH_SIZE, w.get_value(borrow=True).shape[1])

    @property
    def inputs(self):
        return [self.x, self.k]

    @property
    def cost(self):
        err = self.y - self.k
        return TT.mean((err * err).sum(axis=1))


class Classifier(Network):
    '''A classifier attempts to match a 1-hot target output.'''

    def __init__(self, *args, **kwargs):
        self.k = TT.ivector('k')
        super(Classifier, self).__init__(*args, **kwargs)
        self.y = self.softmax(self.y)
        # for shape debugging
        self.k.tag.test_value = (
            3 * numpy.random.randn(DEBUG_BATCH_SIZE)).astype('int32')

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

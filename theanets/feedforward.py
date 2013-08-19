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

import lmj.cli
import pickle
import gzip
import numpy as np
import theano
import theano.tensor as TT

from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

logging = lmj.cli.get_logger(__name__)

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

    def __init__(self, layers, activation, rng=None, input_noise=0,
                 hidden_noise=0, input_dropouts=0, hidden_dropouts=0,
                 **kwargs):
        '''Create a new feedforward network of a specific topology.

        layers: A sequence of integers specifying the number of units at each
          layer. As an example, layers=(10, 20, 3) has one "input" layer with 10
          units, one "hidden" layer with 20 units, and one "output" layer with 3
          units. That is, inputs should be of length 10, and outputs will be of
          length 3.
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

        Available keyword arguments:

        decode: Any of the hidden layers can be tapped at the output. Just
          specify a value greater than 1 to tap the last N hidden layers.
        tied_weights: Construct decoding weights using the transpose of the
          encoding weights on corresponding layers. If not true, decoding
          weights will be constructed using a separate weight matrix.
        '''
        self.hiddens = []
        self.weights = []
        self.biases = []

        self.rng = rng or RandomStreams()
        tied_weights = kwargs.get('tied_weights')
        decode = kwargs.get('decode', 1)

        # in this module, x refers to a network's input, and y to its output.
        self.x = TT.matrix('x')
        self.x.tag.test_value = np.random.randn(DEBUG_BATCH_SIZE, layers[0])

        parameter_count = 0
        sizes = layers[:-1]

        if tied_weights:
            error = 'with --tied-weights, --layers must be a palindrome of length 2k+1'
            assert len(layers) % 2 == 1, error
            k = len(layers) // 2
            encode = np.asarray(layers[:k])
            decode = np.asarray(layers[k+1:])
            assert np.allclose(encode - decode[::-1], 0), error
            sizes = layers[:k+1]

        z = self._noise_and_dropout(self.x, input_noise, input_dropouts)
        for i, (a, b) in enumerate(zip(sizes[:-1], sizes[1:])):
            Wi, bi, params = self._weights_and_bias(a, b, i)
            parameter_count += params
            self.hiddens.append(self._noise_and_dropout(
                activation(TT.dot(z, Wi) + bi), hidden_noise, hidden_dropouts))
            self.weights.append(Wi)
            self.biases.append(bi)
            z = self.hiddens[-1]
            z.tag.test_value = np.random.randn(DEBUG_BATCH_SIZE, b)

        w = len(self.weights)
        if tied_weights:
            for i in range(w - 1, -1, -1):
                h = self.hiddens[-1]
                a, b = self.weights[i].get_value(borrow=True).shape
                logging.info('tied weights from layer %d: %s x %s', i, b, a)
                self.hiddens.append(TT.dot(h - TT.neq(h, 0) * self.biases[i], self.weights[i].T))
        else:
            n = layers[-1]
            decoders = []
            for i in range(w - 1, w - 1 - decode, -1):
                b = self.biases[i].get_value(borrow=True).shape[0]
                Di, _, params = self._weights_and_bias(b, n, 'out_%d' % i)
                parameter_count += params - n
                decoders.append(TT.dot(self.hiddens[i], Di))
                self.weights.append(Di)
            parameter_count += n
            bias = theano.shared(np.zeros((n, ), FLOAT), name='bias_out')
            bias.tag.test_value = np.random.randn(n)
            self.biases.append(bias)
            self.hiddens.append(sum(decoders) + bias)

        logging.info('%d total network parameters', parameter_count)

        self.y = self.hiddens.pop()

        self.updates = {}

        # calling this computes a forward pass, returning all layer activations.
        self.forward = theano.function([self.x], self.hiddens + [self.y])

    @property
    def inputs(self):
        return [self.x]

    @property
    def monitors(self):
        return [self.cost] + self.sparsities

    @property
    def sparsities(self):
        return [TT.eq(h, 0).mean() for h in self.hiddens]

    @staticmethod
    def _weights_and_bias(a, b, suffix):
        '''Helper method for creating a layer of weights and bias values.'''
        arr = np.random.randn(a, b) / np.sqrt(a + b)
        weight = theano.shared(arr.astype(FLOAT), name='W_%s' % suffix)
        weight.tag.test_value = np.random.randn(a, b)
        bias = theano.shared(np.zeros((b, ), FLOAT), name='b_%s' % suffix)
        bias.tag.test_value = np.random.randn(b)
        logging.info('weights for layer %s: %s x %s', suffix, a, b)
        return weight, bias, (a + 1) * b

    def _noise_and_dropout(self, x, sigma, rho):
        '''Add noise and dropouts to elements of x as needed.

        x: input array, updated in-place as needed
        sigma: standard deviation of noise to add to x
        rho: fraction of elements of x to set randomly to 0.
        '''
        if sigma > 0 and rho > 0:
            noise = self.rng.normal(size=x.shape, std=sigma)
            mask = self.rng.binomial(size=x.shape, n=1, p=1-rho, dtype=FLOAT)
            return mask * (x + noise)
        if sigma > 0:
            return x + self.rng.normal(size=x.shape, std=sigma)
        if rho > 0:
            mask = self.rng.binomial(size=x.shape, n=1, p=1-rho, dtype=FLOAT)
            return mask * x
        return x

    def params(self, **kwargs):
        params = []
        params.extend(self.weights)
        if kwargs.get('no_learn_biases'):
            pass
        else:
            params.extend(self.biases)
        return params

    def __call__(self, x):
        '''Compute a forward pass of the inputs, returning the net output.'''
        return self.forward(x)[-1]

    def save(self, filename):
        '''Save the parameters of this network to disk.'''
        opener = gzip.open if filename.lower().endswith('.gz') else open
        handle = opener(filename, 'wb')
        pickle.dump(
            dict(weights=[p.get_value().copy() for p in self.weights],
                 biases=[p.get_value().copy() for p in self.biases],
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
        w = self.weights[len(self.biases) - 1]
        self.k.tag.test_value = np.random.randn(
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
        self.k.tag.test_value = (3 * np.random.randn(DEBUG_BATCH_SIZE)).astype('int32')

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

# Copyright (c) 2012-2014 Leif Johnson <leif@leifjohnson.net>
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

import climate
import pickle
import functools
import gzip
import numpy as np
import theano
import theano.tensor as TT

from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

logging = climate.get_logger(__name__)

FLOAT = theano.config.floatX


class Network(object):
    '''The network class encapsulates a fully-connected feedforward net.

    In addition to defining standard functionality for feedforward nets, there
    are also many options for specifying topology and regularization, several of
    which must be provided to the constructor at initialization time.

    Parameters
    ----------
    layers : sequence of int
        A sequence of integers specifying the number of units at each layer. As
        an example, layers=(10, 20, 3) has one "input" layer with 10 units, one
        "hidden" layer with 20 units, and one "output" layer with 3 units. That
        is, inputs should be of length 10, and outputs will be of length 3.

    activation : string
        The name of an activation function to use on hidden network units.

    rng : theano RandomStreams object, optional
        Use a specific Theano random number generator. A new one will be created
        if this is None.

    input_noise : float, optional
        Standard deviation of desired noise to inject into input.

    hidden_noise : float, optional
        Standard deviation of desired noise to inject into hidden unit
        activation output.

    input_dropouts : float in [0, 1], optional
        Proportion of input units to randomly set to 0.

    hidden_dropouts : float in [0, 1], optional
        Proportion of hidden unit activations to randomly set to 0.

    decode : positive int, optional
        Any of the hidden layers can be tapped at the output. Just specify a
        value greater than 1 to tap the last N hidden layers. The default is 1,
        which decodes from just the last layer.

    tied_weights : bool, optional
        Construct decoding weights using the transpose of the encoding weights
        on corresponding layers. If not True, decoding weights will be
        constructed using a separate weight matrix.

    Attributes
    ----------
    weights : list of Theano shared variables
        Theano shared variables containing network connection weights.

    biases : list of Theano shared variables
        Theano shared variables containing biases for hidden and output units.

    hiddens : list of Theano variables
        Computed Theano variables for the state of hidden units in the network.
    '''

    def __init__(self, layers, activation, **kwargs):
        self.layers = tuple(layers)
        self.activation = activation
        self.hiddens = []
        self.weights = []
        self.biases = []

        self.rng = kwargs.get('rng') or RandomStreams()
        self.tied_weights = bool(kwargs.get('tied_weights'))

        # x is a proxy for our network's input, and y for its output.
        self.x = TT.matrix('x')

        activation = self._build_activation(activation)
        if hasattr(activation, '__theanets_name__'):
            logging.info('hidden activation: %s', activation.__theanets_name__)

        # ensure that --layers is compatible with --tied-weights.
        sizes = layers[:-1]
        if self.tied_weights:
            error = 'with --tied-weights, --layers must be an odd-length palindrome'
            assert len(layers) % 2 == 1, error
            k = len(layers) // 2
            encode = np.asarray(layers[:k])
            decode = np.asarray(layers[k+1:])
            assert np.allclose(encode - decode[::-1], 0), error
            sizes = layers[:k+1]

        _, parameter_count = self._create_forward_map(sizes, activation, **kwargs)

        # set up the "decoding" computations from layer activations to output.
        w = len(self.weights)
        if self.tied_weights:
            for i in range(w - 1, -1, -1):
                h = self.hiddens[-1]
                a, b = self.weights[i].get_value(borrow=True).shape
                logging.info('tied weights from layer %d: %s x %s', i, b, a)
                # --tied-weights implies --no-learn-biases (biases are zero).
                self.hiddens.append(TT.dot(h, self.weights[i].T))
        else:
            n = layers[-1]
            decoders = []
            for i in range(w - 1, w - 1 - kwargs.get('decode', 1), -1):
                b = self.biases[i].get_value(borrow=True).shape[0]
                Di, _, count = self._create_layer(b, n, 'out_%d' % i)
                parameter_count += count - n
                decoders.append(TT.dot(self.hiddens[i], Di))
                self.weights.append(Di)
            parameter_count += n
            bias = theano.shared(np.zeros((n, ), FLOAT), name='bias_out')
            self.biases.append(bias)
            self.hiddens.append(sum(decoders) + bias)

        logging.info('%d total network parameters', parameter_count)

        self.y = self.hiddens.pop()
        self.updates = {}

    @property
    def inputs(self):
        '''Return a list of Theano input variables for this network.'''
        return [self.x]

    @property
    def monitors(self):
        '''Generate a sequence of name-value pairs for monitoring the network.
        '''
        yield 'error', self.cost
        for i, h in enumerate(self.hiddens):
            yield 'h{}<0.1'.format(i+1), (abs(h) < 0.1).mean()
            yield 'h{}<0.9'.format(i+1), (abs(h) < 0.9).mean()

    @staticmethod
    def _create_layer(a, b, suffix, sparse=None):
        '''Create a layer of weights and bias values.

        Parameters
        ----------
        a : int
            Number of rows of the weight matrix -- equivalently, the number of
            "input" units that the weight matrix connects.
        b : int
            Number of columns of the weight matrix -- equivalently, the number
            of "output" units that the weight matrix connects.
        suffix : str
            A string suffix to use in the Theano name for the created variables.
            This string will be appended to "W_" (for the weights) and "b_" (for
            the biases) parameters that are created and returned.
        sparse : float in (0, 1)
            If given, ensure that the weight matrix for the layer has only this
            proportion of nonzero entries.

        Returns
        -------
        weight : Theano shared array
            A shared array containing Theano values representing the weights
            connecting each "input" unit to each "output" unit.
        bias : Theano shared array
            A shared array containing Theano values representing the bias
            values on each of the "output" units.
        count : int
            The number of parameters that are included in the returned
            variables.
        '''
        arr = np.random.randn(a, b) / np.sqrt(a + b)
        if sparse is not None:
            arr *= np.random.binomial(n=1, p=sparse, size=(a, b))
        weight = theano.shared(arr.astype(FLOAT), name='W_{}'.format(suffix))
        bias = theano.shared(np.zeros((b, ), FLOAT), name='b_{}'.format(suffix))
        logging.info('weights for layer %s: %s x %s', suffix, a, b)
        return weight, bias, (a + 1) * b

    def _create_forward_map(self, sizes, activation, **kwargs):
        '''Set up a computation graph to map the input to layer activations.

        Parameters
        ----------
        sizes : list of int
            A list of the number of nodes in each feedforward hidden layer.

        activation : callable
            The activation function to use on each feedforward hidden layer.

        input_noise : float, optional
            Standard deviation of desired noise to inject into input.

        hidden_noise : float, optional
            Standard deviation of desired noise to inject into hidden unit
            activation output.

        input_dropouts : float in [0, 1], optional
            Proportion of input units to randomly set to 0.

        hidden_dropouts : float in [0, 1], optional
            Proportion of hidden unit activations to randomly set to 0.

        Returns
        -------
        parameter_count : int
            The number of parameters created in the forward map.
        '''
        parameter_count = 0
        z = self._add_noise(
            self.x,
            kwargs.get('input_noise', 0.),
            kwargs.get('input_dropouts', 0.))
        for i, (a, b) in enumerate(zip(sizes[:-1], sizes[1:])):
            Wi, bi, count = self._create_layer(a, b, i)
            parameter_count += count
            self.hiddens.append(self._add_noise(
                activation(TT.dot(z, Wi) + bi),
                kwargs.get('hidden_noise', 0.),
                kwargs.get('hidden_dropouts', 0.)))
            self.weights.append(Wi)
            self.biases.append(bi)
            z = self.hiddens[-1]
        return z, parameter_count

    def _add_noise(self, x, sigma, rho):
        '''Add noise and dropouts to elements of x as needed.

        Parameters
        ----------
        x : Theano array
            Input array to add noise and dropouts to.
        sigma : float
            Standard deviation of gaussian noise to add to x. If this is 0, then
            no gaussian noise is added to the values of x.
        rho : float, in [0, 1]
            Fraction of elements of x to set randomly to 0. If this is 0, then
            no elements of x are set randomly to 0. (This is also called
            "salt-and-pepper noise" or "dropouts" in the research community.)

        Returns
        -------
        Theano array
            The parameter x, plus additional noise as specified.
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

    def _compile(self):
        '''If needed, compile the Theano function for this network.'''
        if getattr(self, '_compute', None) is None:
            self._compute = theano.function(
                [self.x], self.hiddens + [self.y], updates=self.updates)

    def _build_activation(self, act=None):
        '''Given an activation description, return a callable that implements it.

        Parameters
        ----------
        activation : string
            A string description of an activation function to use.

        Returns
        -------
        callable(float) -> float :
            A callable activation function.
        '''
        def compose(a, b):
            c = lambda z: b(a(z))
            c.__theanets_name__ = '%s(%s)' % (b.__theanets_name__, a.__theanets_name__)
            return c
        if '+' in act:
            return functools.reduce(
                compose, (self._build_activation(a) for a in act.split('+')))
        options = {
            'tanh': TT.tanh,
            'linear': lambda z: z,
            'logistic': TT.nnet.sigmoid,
            'sigmoid': TT.nnet.sigmoid,
            'softplus': TT.nnet.softplus,

            # shorthands
            'relu': lambda z: TT.maximum(0, z),
            'trec': lambda z: z * (z > 1),
            'tlin': lambda z: z * (abs(z) > 1),

            # modifiers
            'rect:max': lambda z: TT.minimum(1, z),
            'rect:min': lambda z: TT.maximum(0, z),

            # normalization
            'norm:dc': lambda z: (z.T - z.mean(axis=1)).T,
            'norm:max': lambda z: (z.T / TT.maximum(1e-10, abs(z).max(axis=1))).T,
            'norm:std': lambda z: (z.T / TT.maximum(1e-10, TT.std(z, axis=1))).T,
            }
        for k, v in options.items():
            v.__theanets_name__ = k
        try:
            return options[act]
        except KeyError:
            raise KeyError('unknown activation %r' % act)

    def params(self, **kwargs):
        '''Return a list of the Theano parameters for this network.'''
        params = []
        params.extend(self.weights)
        if getattr(self, 'tied_weights', False) or kwargs.get('no_learn_biases'):
            # --tied-weights implies --no-learn-biases.
            pass
        else:
            params.extend(self.biases)
        return params

    def feed_forward(self, x):
        '''Compute a forward pass of all activations from the given input.

        Parameters
        ----------
        x : ndarray
            An array containing data to be fed into the network.

        Returns
        -------
        list of ndarray
            Returns the activation values of each layer in the the network when
            given input `x`.
        '''
        self._compile()
        return self._compute(x)

    def predict(self, x):
        '''Compute a forward pass of the inputs, returning the net output.

        Parameters
        ----------
        x : ndarray
            An array containing data to be fed into the network.

        Returns
        -------
        ndarray
            Returns the values of the network output units when given input `x`.
        '''
        return self.feed_forward(x)[-1]

    __call__ = predict

    def save(self, filename):
        '''Save the parameters of this network to disk.

        Parameters
        ----------
        filename : str
            Save the parameters of this network to a pickle file at the named
            path. If this name ends in ".gz" then the output will automatically
            be gzipped; otherwise the output will be a "raw" pickle.
        '''
        opener = gzip.open if filename.lower().endswith('.gz') else open
        handle = opener(filename, 'wb')
        pickle.dump(
            dict(weights=[p.get_value().copy() for p in self.weights],
                 biases=[p.get_value().copy() for p in self.biases],
                 ), handle, -1)
        handle.close()
        logging.info('%s: saved model parameters', filename)

    def load(self, filename):
        '''Load the parameters for this network from disk.

        Parameters
        ----------
        filename : str
            Load the parameters of this network from a pickle file at the named
            path. If this name ends in ".gz" then the input will automatically
            be gunzipped; otherwise the input will be treated as a "raw" pickle.
        '''
        opener = gzip.open if filename.lower().endswith('.gz') else open
        handle = opener(filename, 'rb')
        params = pickle.load(handle)
        for target, source in zip(self.weights, params['weights']):
            target.set_value(source)
        for target, source in zip(self.biases, params['biases']):
            target.set_value(source)
        handle.close()
        logging.info('%s: loaded model parameters', filename)

    def J(self, weight_l1=0, weight_l2=0, hidden_l1=0, hidden_l2=0, contractive_l2=0, **unused):
        '''Return a variable representing the cost or loss for this network.

        Parameters
        ----------
        weight_l1 : float, optional
            Regularize the L1 norm of unit connection weights by this constant.
        weight_l2 : float, optional
            Regularize the L2 norm of unit connection weights by this constant.
        hidden_l1 : float, optional
            Regularize the L1 norm of hidden unit activations by this constant.
        hidden_l2 : float, optional
            Regularize the L2 norm of hidden unit activations by this constant.
        contractive_l2 : float, optional
            Regularize model using the Frobenius norm of the hidden Jacobian.

        Returns
        -------
        Theano variable
            A variable representing the overall cost value of this network.
        '''
        cost = self.cost
        if weight_l1 > 0:
            cost += weight_l1 * sum(abs(w).sum() for w in self.weights)
        if weight_l2 > 0:
            cost += weight_l2 * sum((w * w).sum() for w in self.weights)
        if hidden_l1 > 0:
            cost += hidden_l1 * sum(abs(h).mean(axis=0).sum() for h in self.hiddens)
        if hidden_l2 > 0:
            cost += hidden_l2 * sum((h * h).mean(axis=0).sum() for h in self.hiddens)
        if contractive_l2 > 0:
            cost += contractive_l2 * sum(
                TT.sqr(TT.grad(h.mean(axis=0).sum(), self.x)).sum() for h in self.hiddens)
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

    @staticmethod
    def softmax(x):
        # TT.nnet.softmax doesn't work with the HF trainer.
        z = TT.exp(x.T - x.T.max(axis=0))
        return (z / z.sum(axis=0)).T

    @property
    def inputs(self):
        return [self.x, self.k]

    @property
    def cost(self):
        return -TT.mean(TT.log(self.y)[TT.arange(self.k.shape[0]), self.k])

    @property
    def incorrect(self):
        return TT.mean(TT.neq(TT.argmax(self.y, axis=1), self.k))

    @property
    def monitors(self):
        yield 'incorrect', self.incorrect
        for i, h in enumerate(self.hiddens):
            yield 'h{}<0.1'.format(i+1), (abs(h) < 0.1).mean()
            yield 'h{}<0.9'.format(i+1), (abs(h) < 0.9).mean()

    def classify(self, x):
        return self.predict(x).argmax(axis=1)

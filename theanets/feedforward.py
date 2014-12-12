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
import functools
import gzip
import numpy as np
import pickle
import theano
import theano.tensor as TT
import warnings

from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

logging = climate.get_logger(__name__)

FLOAT = theano.config.floatX


def load(filename, **kwargs):
    '''Load an entire network from a pickle file on disk.

    Parameters
    ----------
    filename : str
        Load the keyword arguments and parameters of a network from a pickle
        file at the named path. If this name ends in ".gz" then the input will
        automatically be gunzipped; otherwise the input will be treated as a
        "raw" pickle.

    Returns
    -------
    Network :
        A newly-constructed network, with topology and parameters loaded from
        the given pickle file.
    '''
    opener = gzip.open if filename.lower().endswith('.gz') else open
    handle = opener(filename, 'rb')
    pkl = pickle.load(handle)
    handle.close()
    kw = pkl['kwargs']
    kw.update(kwargs)
    net = pkl['klass'](**kw)
    net.load_params(filename)
    return net


def softmax(x):
    # TT.nnet.softmax doesn't work with the HF trainer.
    z = TT.exp(x.T - x.T.max(axis=0))
    return (z / z.sum(axis=0)).T


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

    hidden_activation : str, optional
        The name of an activation function to use on hidden network units.
        Defaults to 'sigmoid'.

    output_activation : str, optional
        The name of an activation function to use on output units. Defaults to
        'linear'.

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

    decode_from : positive int, optional
        Any of the hidden layers can be tapped at the output. Just specify a
        value greater than 1 to tap the last N hidden layers. The default is 1,
        which decodes from just the last layer.

    tied_weights : bool, optional
        Construct decoding weights using the transpose of the encoding weights
        on corresponding layers. Defaults to False, which means decoding weights
        will be constructed using a separate weight matrix.

    Attributes
    ----------
    weights : list of Theano shared variables
        Theano shared variables containing network connection weights.

    biases : list of Theano shared variables
        Theano shared variables containing biases for hidden and output units.

    hiddens : list of Theano variables
        Computed Theano variables for the state of hidden units in the network.

    preacts : list of Theano variables
        Computed Theano variables representing the pre-activation inputs for
        network units.

    kwargs : dict
        A dictionary containing the keyword arguments used to construct the
        network.
    '''

    def __init__(self, **kwargs):
        self.preacts = []
        self.hiddens = []
        self.weights = []
        self.biases = []
        self.updates = {}
        self.kwargs = kwargs
        self.rng = kwargs.get('rng') or RandomStreams()

        self.hidden_activation = kwargs.get('hidden_activation', 'logistic')
        self._hidden_func = self._build_activation(self.hidden_activation)
        if hasattr(self._hidden_func, '__theanets_name__'):
            logging.info('hidden activation: %s', self._hidden_func.__theanets_name__)

        self.output_activation = kwargs.get('output_activation', 'linear')
        self._output_func = self._build_activation(self.output_activation)
        if hasattr(self._output_func, '__theanets_name__'):
            logging.info('output activation: %s', self._output_func.__theanets_name__)

        self.setup_vars()
        self.setup_layers(**kwargs)

    def setup_vars(self):
        '''Setup Theano variables for our network.'''
        # x is a proxy for our network's input, and y for its output.
        self.x = TT.matrix('x')

    def setup_layers(self, **kwargs):
        '''Set up a computation graph for our network.

        Parameters
        ----------
        input_noise : float, optional
            Standard deviation of desired noise to inject into input.
        hidden_noise : float, optional
            Standard deviation of desired noise to inject into hidden unit
            activation output.
        input_dropouts : float in [0, 1], optional
            Proportion of input units to randomly set to 0.
        hidden_dropouts : float in [0, 1], optional
            Proportion of hidden unit activations to randomly set to 0.
        decode_from : int, optional
            Compute the activation of the output vector using the activations of
            the last N hidden layers in the network. Defaults to 1, which
            results in a traditional setup that decodes only from the
            penultimate layer in the network.
        '''
        count = 0

        # add noise to inputs.
        x = self._add_noise(self.x,
                            kwargs.get('input_noise', 0),
                            kwargs.get('input_dropouts', 0))

        # setup "encoder" layers.
        kw = dict(
            noise=kwargs.get('hidden_noise', 0),
            dropout=kwargs.get('hidden_dropouts', 0),
        )
        sizes = self.get_encoder_layers()
        for i, (nin, nout) in enumerate(zip(sizes[:-1], sizes[1:])):
            z = self.hiddens and self.hiddens[-1] or x
            count += self.add_feedforward_layer(z, nin, nout, label=i, **kw)

        count += self.setup_decoder(**kwargs)

        logging.info('%d total network parameters', count)

    def add_feedforward_layer(self, input, nin, nout, **kwargs):
        '''Add a new feedforward layer to the network.

        Parameters
        ----------
        input : theano variable
            The theano variable that represents the inputs to this layer.
        nin : int
            The number of input units to this layer.
        nout : out
            The number of output units from this layer.
        label : any, optional
            The name of this layer, used for logging and as the theano variable
            name suffix. Defaults to the index of this layer in the network.
        noise : float, optional
            Add zero-mean gaussian noise to the output activation of hidden
            units. Defaults to 0 (no noise).
        dropout : float, optional
            Simulate "dropout" in this layer by setting the given fraction of
            output activations randomly to zero. Defaults to 0 (no dropout).

        Returns
        -------
        count : int
            A count of the number of learnable parameters in this layer.
        '''
        label = kwargs.get('label') or len(self.hiddens)
        W, nw = self.create_weights(nin, nout, label)
        b, nb = self.create_bias(nout, label)
        pre = TT.dot(input, W) + b
        out = self._add_noise(self._hidden_func(pre),
                              kwargs.get('noise', 0),
                              kwargs.get('dropout', 0))
        self.weights.append(W)
        self.biases.append(b)
        self.preacts.append(pre)
        self.hiddens.append(out)
        return nw + nb

    def setup_decoder(self, **kwargs):
        '''Set up the "decoding" computations from layer activations to output.

        Parameters
        ----------
        decode_from : int, optional
            Compute the activation of the output vector using the activations of
            the last N hidden layers in the network. Defaults to 1, which
            results in a traditional setup that decodes only from the
            penultimate layer in the network.

        Returns
        -------
        count : int
            The number of parameters created in the decoding map.
        '''
        count = 0
        B = len(self.biases) - 1
        nout = self.layers[-1]
        decoders = []
        for i in range(B, B - kwargs.get('decode_from', 1), -1):
            nin = self.biases[i].get_value(borrow=True).shape[0]
            W, n = self.create_weights(nin, nout, 'out_%d' % i)
            count += n
            decoders.append(TT.dot(self.hiddens[i], W))
            self.weights.append(W)
        bias = theano.shared(np.zeros((nout, ), FLOAT), name='bias_out')
        count += nout
        pre = sum(decoders) + bias
        self.biases.append(bias)
        self.preacts.append(pre)
        self.y = self._output_func(pre)
        return count

    def get_encoder_layers(self):
        '''Compute the layers that will be part of the network encoder.

        Returns
        -------
        layers : list of int
            A list of integers specifying sizes of the encoder network layers.
        '''
        return self.layers[:-1]

    @property
    def inputs(self):
        '''Return a list of Theano input variables for this network.'''
        return [self.x]

    @property
    def monitors(self):
        '''Generate a sequence of name-value pairs for monitoring the network.
        '''
        yield 'err', self.cost
        for i, h in enumerate(self.hiddens):
            yield 'h{}<0.1'.format(i+1), 100 * (abs(h) < 0.1).mean()
            yield 'h{}<0.9'.format(i+1), 100 * (abs(h) < 0.9).mean()

    @property
    def layers(self):
        '''A tuple containing the layer configuration for this network.'''
        return self.kwargs['layers']

    @property
    def tied_weights(self):
        '''A boolean indicating this network uses tied weights.'''
        return self.kwargs.get('tied_weights', False)

    @staticmethod
    def create_weights(a, b, suffix, **kwargs):
        '''Create a layer of randomly-initialized weights.

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
            This string will be appended to 'W_' to name the parameters that are
            created and returned.
        sparse : float in (0, 1), optional
            If given, ensure that the given fraction of the weight matrix is
            set to zero. Defaults to 0, meaning all weights are nonzero.
        radius : float, optional
            If given, rescale the initial weights to have this spectral radius.
            No scaling is performed by default.

        Returns
        -------
        weight : Theano shared array
            A shared array containing Theano values representing the weights
            connecting each "input" unit to each "output" unit.
        count : int
            The number of parameters that are included in the returned
            variables.
        '''
        arr = np.random.randn(a, b) / np.sqrt(a + b)
        sparse = kwargs.get('sparse')
        if sparse and 0 < sparse < 1:
            k = min(a, b)
            mask = np.random.binomial(n=1, p=1 - sparse, size=(a, b)).astype(bool)
            mask[:k, :k] |= np.random.permutation(np.eye(k).astype(bool))
            arr *= mask
        radius = kwargs.get('radius')
        if radius:
            # rescale weights to have the appropriate spectral radius.
            u, s, vT = np.linalg.svd(arr)
            arr = np.dot(np.dot(u, np.diag(radius * s / abs(s[0]))), vT)
        weight = theano.shared(arr.astype(FLOAT), name='W_{}'.format(suffix))
        logging.info('weights for layer %s: %s x %s', suffix, a, b)
        return weight, a * b

    @staticmethod
    def create_bias(b, suffix):
        '''Create a vector of bias values.

        Parameters
        ----------
        b : int
            Number of units of bias to create.
        suffix : str
            A string suffix to use in the Theano name for the created variables.
            This string will be appended to 'b_' to name the parameters that are
            created and returned.

        Returns
        -------
        bias : Theano shared array
            A shared array containing Theano values representing the bias for a
            set of computation units.
        count : int
            The number of parameters that are included in the returned
            variables.
        '''
        arr = 1e-6 * np.random.randn(b)
        bias = theano.shared(arr.astype(FLOAT), name='b_{}'.format(suffix))
        return bias, b

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
            "masking noise" (for inputs) or "dropouts" (for hidden units).)

        Returns
        -------
        Theano array
            The parameter x, plus additional noise as specified.
        '''
        if sigma > 0 and rho > 0:
            noise = self.rng.normal(size=x.shape, std=sigma, dtype=FLOAT)
            mask = self.rng.binomial(size=x.shape, n=1, p=1-rho, dtype=FLOAT)
            return mask * (x + noise)
        if sigma > 0:
            return x + self.rng.normal(size=x.shape, std=sigma, dtype=FLOAT)
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
            'softmax': softmax,

            # shorthands
            'relu': lambda z: z * (z > 0),
            'trel': lambda z: z * (z > 0) * (z < 1),
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
        if not kwargs.get('no_learn_biases'):
            params.extend(self.biases)
        return params

    def get_weights(self, layer, borrow=False):
        '''Return the current weights for a given layer.

        Parameters
        ----------
        layer : int
            The layer of weights to return.
        borrow : bool, optional
            Whether to "borrow" the reference to the weights. If True, this
            returns a view onto the current weight array; if False (default), it
            returns a copy of the weight array.

        Returns
        -------
        ndarray :
            The weight values, as a numpy array.
        '''
        return self.weights[layer].get_value(borrow=borrow)

    def get_biases(self, layer, borrow=False):
        '''Return the current bias vector for a given layer.

        Parameters
        ----------
        layer : int
            The layer of bias values to return.
        borrow : bool, optional
            Whether to "borrow" the reference to the biases. If True, this
            returns a view onto the current bias vector; if False (default), it
            returns a copy of the biases.

        Returns
        -------
        ndarray :
            The bias values, as a numpy vector.
        '''
        return self.biases[layer].get_value(borrow=borrow)

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
        '''Save the state of this network to a pickle file on disk.

        Parameters
        ----------
        filename : str
            Save the parameters of this network to a pickle file at the named
            path. If this name ends in ".gz" then the output will automatically
            be gzipped; otherwise the output will be a "raw" pickle.
        '''
        opener = gzip.open if filename.lower().endswith('.gz') else open
        handle = opener(filename, 'wb')
        pickle.dump(dict(
            weights=[p.get_value().copy() for p in self.weights],
            biases=[p.get_value().copy() for p in self.biases],
            klass=self.__class__, kwargs=self.kwargs), handle, -1)
        handle.close()
        logging.info('%s: saved model parameters', filename)

    def load_params(self, filename):
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
        saved = pickle.load(handle)
        for target, source in zip(self.weights, saved['weights']):
            logging.info('%s: setting value %s', target.name, source.shape)
            target.set_value(source)
        for target, source in zip(self.biases, saved['biases']):
            logging.info('%s: setting value %s', target.name, source.shape)
            target.set_value(source)
        handle.close()
        logging.info('%s: loaded model parameters', filename)

    def load(self, filename):
        warnings.warn(
            'please use Network.load_params instead of Network.load',
            DeprecationWarning)
        return self.load_params(filename)

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

    def setup_decoder(self, **kwargs):
        '''Set up weights for the decoder layers of an autoencoder.

        This implementation allows for weights to be tied to encoder weights.

        Parameters
        ----------
        input_noise : float, optional
            Standard deviation of desired noise to inject into input.
        hidden_noise : float, optional
            Standard deviation of desired noise to inject into hidden unit
            activation output.
        input_dropouts : float in [0, 1], optional
            Proportion of input units to randomly set to 0.
        hidden_dropouts : float in [0, 1], optional
            Proportion of hidden unit activations to randomly set to 0.
        tied_weights : bool, optional
            If True, use decoding weights that are "tied" to the encoding
            weights. This only makes sense for a limited set of "autoencoder"
            layer configurations. Defaults to False.
        decode_from : int, optional
            For networks without tied weights, compute the activation of the
            output vector using the activations of the last N hidden layers in
            the network. Defaults to 1, which results in a traditional setup
            that decodes only from the penultimate layer in the network.

        Returns
        -------
        count : int
            A count of the number of tunable decoder parameters.
        '''
        if not self.tied_weights:
            return super(Autoencoder, self).setup_decoder(**kwargs)
        count = 0
        noise = kwargs.get('hidden_noise', 0)
        dropout = kwargs.get('hidden_dropouts', 0)
        for i in range(len(self.weights) - 1, -1, -1):
            nin, nout = self.weights[i].get_value(borrow=True).shape
            logging.info('tied weights from layer %d: %s x %s', i, nout, nin)
            bias = theano.shared(np.zeros((nin, ), FLOAT), name='b_out{}'.format(i))
            count += nin
            pre = TT.dot(self.hiddens[-1], self.weights[i].T) + bias
            act = self._add_noise(self._hidden_func(pre), noise, dropout)
            if i == 0:
                act = self._output_func(pre)
            self.biases.append(bias)
            self.preacts.append(pre)
            self.hiddens.append(act)
        self.y = self.hiddens.pop()
        return count

    def get_encoder_layers(self):
        '''Compute the layers that will be part of the network encoder.

        This implementation ensures that --layers is compatible with
        --tied-weights.

        Returns
        -------
        layers : list of int
            A list of integers specifying sizes of the encoder network layers.
        '''
        sizes = self.layers[:-1]
        if self.tied_weights:
            error = 'with --tied-weights, --layers must be an odd-length palindrome'
            assert len(self.layers) % 2 == 1, error
            k = len(self.layers) // 2
            encode = np.asarray(self.layers[:k])
            decode = np.asarray(self.layers[k+1:])
            assert (encode == decode[::-1]).all(), error
            sizes = self.layers[:k+1]
        return sizes

    @property
    def cost(self):
        err = self.y - self.x
        return TT.mean((err * err).sum(axis=1))

    def encode(self, x, layer=None, sample=False):
        '''Encode a dataset using the hidden layer activations of our network.

        Parameters
        ----------
        x : ndarray
            A dataset to encode. Rows of this dataset capture individual data
            points, while columns represent the variables in each data point.

        layer : int, optional
            The index of the hidden layer activation to use. By default, we use
            the "middle" hidden layer---for example, for a 4,2,4 or 4,3,2,3,4
            autoencoder, we use the "2" layer (index 1 or 2, respectively).

        sample : bool, optional
            If True, then draw a sample using the hidden activations as
            independent Bernoulli probabilities for the encoded data. This
            assumes the hidden layer has a logistic sigmoid activation function.

        Returns
        -------
        ndarray :
            The given dataset, encoded by the appropriate hidden layer
            activation.
        '''
        enc = self.feed_forward(x)[(layer or len(self.layers) // 2) - 1]
        if sample:
            return np.random.binomial(n=1, p=enc).astype(np.uint8)
        return enc

    def decode(self, z, layer=None):
        '''Decode an encoded dataset by computing the output layer activation.

        Parameters
        ----------
        z : ndarray
            A matrix containing encoded data from this autoencoder.

        layer : int, optional
            The index of the hidden layer that was used to encode `z`.

        Returns
        -------
        ndarray :
            The decoded dataset.
        '''
        if not hasattr(self, '_decoders'):
            self._decoders = {}
        layer = layer or len(self.layers) // 2
        if layer not in self._decoders:
            self._decoders[layer] = theano.function(
                [self.hiddens[layer - 1]], [self.y], updates=self.updates)
        return self._decoders[layer](z)[0]


class Regressor(Network):
    '''A regressor attempts to produce a target output.'''

    def setup_vars(self):
        super(Regressor, self).setup_vars()

        # the k variable holds the target output for input x.
        self.k = TT.matrix('k')

    @property
    def inputs(self):
        return [self.x, self.k]

    @property
    def cost(self):
        err = self.y - self.k
        return TT.mean((err * err).sum(axis=1))


class Classifier(Network):
    '''A classifier attempts to match a 1-hot target output.'''

    def __init__(self, **kwargs):
        kwargs['output_activation'] = 'softmax'
        super(Classifier, self).__init__(**kwargs)

    def setup_vars(self):
        super(Classifier, self).setup_vars()

        # for a classifier, k specifies the correct labels for a given input.
        self.k = TT.ivector('k')

    @property
    def inputs(self):
        return [self.x, self.k]

    @property
    def cost(self):
        return -TT.mean(TT.log(self.y)[TT.arange(self.k.shape[0]), self.k])

    @property
    def accuracy(self):
        '''Compute the percent correct classifications.'''
        return 100 * TT.mean(TT.eq(TT.argmax(self.y, axis=1), self.k))

    @property
    def monitors(self):
        yield 'acc', self.accuracy
        for i, h in enumerate(self.hiddens):
            yield 'h{}<0.1'.format(i+1), 100 * (abs(h) < 0.1).mean()
            yield 'h{}<0.9'.format(i+1), 100 * (abs(h) < 0.9).mean()

    def classify(self, x):
        return self.predict(x).argmax(axis=1)

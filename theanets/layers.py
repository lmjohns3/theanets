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

'''This module contains classes for different types of network layers.'''

import climate
import functools
import numpy as np
import sys
import theano
import theano.tensor as TT

from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

logging = climate.get_logger(__name__)

FLOAT = theano.config.floatX


def create_matrix(nin, nout, name, sparse=0, radius=0):
    '''Create a matrix of randomly-initialized weights.

    Parameters
    ----------
    nin : int
        Number of rows of the weight matrix -- equivalently, the number of
        "input" units that the weight matrix connects.
    nout : int
        Number of columns of the weight matrix -- equivalently, the number
        of "output" units that the weight matrix connects.
    name : str
        A string to use as the theano name for the created variable.
    sparse : float in (0, 1), optional
        If given, ensure that the given fraction of the weight matrix is
        set to zero. Defaults to 0, meaning all weights are nonzero.
    radius : float, optional
        If given, rescale the initial weights to have this spectral radius.
        No scaling is performed by default.

    Returns
    -------
    matrix : theano shared array
        A shared array containing a matrix of theano values. These often
        represent the weights connecting each "input" unit to each "output" unit
        in a layer.
    '''
    arr = np.random.randn(nin, nout) / np.sqrt(nin + nout)
    if 0 < sparse < 1:
        k = min(nin, nout)
        mask = np.random.binomial(n=1, p=1 - sparse, size=(nin, nout)).astype(bool)
        mask[:k, :k] |= np.random.permutation(np.eye(k).astype(bool))
        arr *= mask
    if radius:
        # rescale weights to have the appropriate spectral radius.
        u, s, vT = np.linalg.svd(arr)
        arr = np.dot(np.dot(u, np.diag(radius * s / abs(s[0]))), vT)
    return theano.shared(arr.astype(FLOAT), name=name)


def create_vector(size, name):
    '''Create a vector of small values.

    Parameters
    ----------
    size : int
        Length of vecctor to create.
    name : str
        A string to use as the theano name for the created variables.

    Returns
    -------
    vector : theano shared array
        A shared array containing a vector of theano values. This often
        represents the bias for a layer of computation units.
    '''
    return theano.shared((1e-6 * np.random.randn(size)).astype(FLOAT), name=name)


def softmax(x):
    '''Compute the softmax of the rows of a matrix x.

    Parameters
    ----------
    x : theano variable
        A theano matrix. Each row represents one data point, and each column
        represents one of the possible classes for the data points.

    Returns
    -------
    y : theano variable
        A theano expression computing the softmax of each row of `x`.
    '''
    # TT.nnet.softmax doesn't work with the HF trainer.
    z = TT.exp(x.T - x.T.max(axis=0))
    return (z / z.sum(axis=0)).T


def create_activation(activation):

    '''Given an activation description, return a callable that implements it.

    Parameters
    ----------
    activation : string
        A string description of an activation function to use.

    Returns
    -------
    activation : callable(float) -> float
        A callable activation function.
    '''
    def compose(a, b):
        c = lambda z: b(a(z))
        c.__theanets_name__ = '%s(%s)' % (b.__theanets_name__, a.__theanets_name__)
        return c
    if '+' in activation:
        return functools.reduce(
            compose, (create_activation(a) for a in activation.split('+')))
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
        return options[activation.lower()]
    except KeyError:
        raise KeyError('unknown activation {}'.format(activation))


def add_noise(input, level, rng):
    '''Add noise to elements of the input variable as needed.

    Parameters
    ----------
    input : theano variable
        Input variable to add noise to.
    level : float
        Standard deviation of gaussian noise to add to the input. If this is
        0, then no gaussian noise is added to the input.

    Returns
    -------
    output : theano variable
        The input variable, plus additional noise as specified.
    '''
    if level == 0:
        return input
    return input + rng.normal(size=input.shape, std=level, dtype=FLOAT)


def add_dropout(input, probability, rng):
    '''Add dropouts to elements of the input variable as needed.

    Parameters
    ----------
    input : theano variable
        Input variable to add dropouts to.
    probability : float, in [0, 1]
        Probability of dropout for each element of the input. If this is 0,
        then no elements of the input are set randomly to 0.

    Returns
    -------
    output : theano variable
        The input variable, plus additional dropouts as specified.
    '''
    if probability == 0:
        return input
    return input * rng.binomial(size=input.shape, n=1, p=1-probability, dtype=FLOAT)


class Registrar(type):
    '''A metaclass that builds a registry of its subclasses.'''

    def __init__(cls, name, bases, dct):
        if not hasattr(cls, '_registry'):
            cls._registry = {}
        else:
            cls._registry[name.lower()] = cls
        super(Registrar, cls).__init__(name, bases, dct)

    def build(cls, name, *args, **kwargs):
        return cls._registry[name.lower()](*args, **kwargs)

# py2k and py3k have different metaclass syntax. :-/
if sys.version_info.major <= 2:
    class Base(object):
        __metaclass__ = Registrar
else:
    class Base(metaclass=Registrar):
        pass


class Layer(Base):
    '''Layers in network graphs derive from this base class.

    In ``theanets``, a layer refers to a set of weights and biases, plus the
    "output" units that produce some sort of signal for further layers to
    consume. The first layer in a network, the input layer, is a special case
    with linear activation and no weights or bias.

    Parameters
    ----------
    nin : int
    nout : int
    name : str, optional
    rng : random number generator, optional
    noise : positive float, optional
    dropout : float in (0, 1), optional
    activation : str, optional
    '''

    count = 0

    def __init__(self, **kwargs):
        '''
        '''
        Layer.count += 1
        self._rng = kwargs.get('rng', RandomStreams())
        self.name = kwargs.get('name', 'l{}'.format(Layer.count))
        self.nin = kwargs['nin']
        self.nout = kwargs['nout']
        self.noise = kwargs.get('noise', 0)
        self.dropout = kwargs.get('dropout', 0)
        self.activation = kwargs.get('activation', 'logistic')
        self.activate = create_activation(self.activation)
        self.weights = []
        self.biases = []

    def output(self, *inputs):
        '''
        '''
        clean = self.activate(self.transform(*inputs))
        noisy = add_noise(clean, self.noise, self._rng)
        return add_dropout(noisy, self.dropout, self._rng)

    def transform(self, *inputs):
        '''
        '''
        return inputs[0]

    def reset(self, **kwargs):
        '''Reset the state of this layer to a new initial condition.

        Returns
        -------
        count : int
            A count of the number of parameters in this layer.
        '''
        return 0

    def get_params(self, exclude_bias=False):
        '''Get a list of parameters in this layer that can be optimized.

        Parameters
        ----------
        exclude_bias : bool, optional
            If True, then do not include bias parameters in this list. Defaults
            to False, which includes both weights and bias parameters.

        Returns
        -------
        params : list of theano shared variables
            A list of the parameters in this layer that can be optimized.
        '''
        return self.weights if exclude_bias else self.weights + self.biases

    def get_regularizers(self, **kwargs):
        '''
        '''
        return None

    def get_monitors(self):
        '''Get monitors for this layer.

        Returns
        -------
        monitors : list of (name, theano expression) pairs
            A list of pairs that represent quantities to monitor during
            training. The first item in the pair is the name of the expression,
            and the second is a theano expression for computing the monitor
            value.
        '''
        return [
            (self._fmt('{}<0.1'), 100 * (abs(self.output) < 0.1).mean()),
            (self._fmt('{}<0.9'), 100 * (abs(self.output) < 0.9).mean()),
        ]

    def _fmt(self, string):
        '''Helper method to format our name into a string.'''
        return string.format(self.name)


class Input(Layer):
    '''The input of a network is a special type of layer with no parameters.

    Input layers essentially add only noise to the input data (if desired), but
    otherwise reproduce their inputs exactly.
    '''

    def __init__(self, size, **kwargs):
        '''
        '''
        kw = {}
        kw.update(**kwargs)
        kw.update(dict(nin=0, nout=size, activation='linear'))
        super(Layer, self).__init__(**kw)


class Feedforward(Layer):
    '''
    '''

    def transform(self, *inputs):
        '''
        '''
        assert len(inputs) == 1
        return TT.dot(inputs[0], self.weights[0]) + self.biases[0]

    def reset(self, **kwargs):
        '''Reset the state of this layer to a new initial condition.

        Returns
        -------
        count : int
            A count of the number of parameters in this layer.
        '''
        logging.info('initializing layer %s: %s x %s', self.name, self.nin, self.nout)
        self.weights = [create_matrix(
            self.nin, self.nout, self._fmt('weights_{}'), **kwargs)]
        self.biases = [create_vector(self.nout, self._fmt('bias_{}'))]
        return self.nout * (self.nin + 1)


class Tied(Feedforward):
    '''
    '''

    def __init__(self, partner, **kwargs):
        '''
        '''
        self.partner = partner
        kw = {}
        kw.update(**kwargs)
        kw.update(nin=partner.nout, nout=partner.nin)
        super(Tied, self).__init__(**kw)

    def transform(self, *inputs):
        assert len(inputs) == 1
        return TT.dot(inputs[0], self.partner.weights[0].T) + self.biases[0]

    def reset(self, **kwargs):
        logging.info('initializing layer %s: %s x %s', self.name, self.nin, self.nout)
        self.biases = [create_vector(self.nout, self._fmt('bias_{}'))]
        return self.nout


class Recurrent(Layer):
    '''
    '''

    def __init__(self, batch_size=64, **kwargs):
        '''
        '''
        super(Recurrent).__init__(**kwargs)
        zeros = np.zeros((batch_size, self.nout), FLOAT)
        self.h_0 = theano.shared(zeros, name=self._fmt('0_{}'))

    def reset(self, **kwargs):
        '''Add a new recurrent layer to the network.

        Parameters
        ----------
        sparse : float in (0, 1), optional
            If given, create sparse connections in the recurrent weight matrix,
            such that this fraction of the weights is set to zero. By default,
            this parameter is 0, meaning all recurrent  weights are nonzero.
        radius : float, optional
            If given, rescale the initial weights to have this spectral radius.
            No scaling is performed by default.

        Returns
        -------
        count : int
            The number of learnable parameters in this layer.
        '''
        logging.info('initializing recurrent layer %s: %s x %s',
                     self.name, self.nin, self.nout)
        self.weights = [
            create_matrix(nin, nout, self._fmt('xh_{}')),
            create_matrix(nout, nout, self._fmt('hh_{}'), **kwargs),
        ]
        self.biases = [create_vector(self.nout, self._fmt('bias_{}'))]
        return self.nout * (1 + self.nin + self.nout)

    def transform(self, *inputs):
        '''
        '''
        assert len(inputs) == 1
        def fn(x_t, h_tm1, W_xh, W_hh, b_h):
            return self.activate(TT.dot(x_t, W_xh) + TT.dot(h_tm1, W_hh) + b_h)
        return self.scan(self._fmt('rnn_{}'), fn, inputs[0])

    def output(self, *inputs):
        '''
        '''
        clean, updates = self.transform(*inputs)
        noisy = add_noise(clean, self.noise, self._rng)
        return add_dropout(noisy, self.dropout, self._rng), updates

    def scan(self, name, fn, *inputs):
        '''
        '''
        return theano.scan(
            name=name, fn=fn, sequences=inputs,
            non_sequences=self.weights + self.biases,
            outputs_info=[self.h_0])


class MRNN(Recurrent):
    '''Define recurrent network layers using multiplicative dynamics.

    The formulation of MRNN implemented here uses a factored dynamics matrix as
    described in Sutskever, Martens & Hinton, ICML 2011, "Generating text with
    recurrent neural networks." This paper is available online at
    http://www.icml-2011.org/papers/524_icmlpaper.pdf.
    '''

    def __init__(self, factors=None, **kwargs):
        self.factors = factors or int(np.ceil(np.sqrt(kwargs['nout'])))
        super(MRNN, self).__init__(**kwargs)

    def reset(self, **kwargs):
        '''Reset the weights and biases for this layer to random values.

        Returns
        -------
        count : int
            The number of learnable parameters in this layer.
        '''
        logging.info('initializing mrnn layer %s: %s x %s [%s]',
                     self.name, self.nin, self.nout, self.factors)
        self.weights = [
            create_matrix(nin, nout, self._fmt('xh_{}')),
            create_matrix(nin, factors, self._fmt('xf_{}')),
            create_matrix(nout, factors, self._fmt('hf_{}')),
            create_matrix(factors, nout, self._fmt('fh_{}')),
        ]
        self.biases = [create_vector(self.nout, self._fmt('bias_{}'))]
        return self.nout * (1 + self.nin) + self.factors * (2 * self.nout + self.nin)

    def transform(self, *inputs):
        '''
        '''
        assert len(inputs) == 1
        def fn(x_t, h_tm1, W_xh, W_xf, W_hf, W_fh, b_h):
            f_t = TT.dot(TT.dot(h_tm1, W_hf) * TT.dot(x_t, W_xf), W_fh)
            return self.activate(TT.dot(x_t, W_xh) + b_h + f_t)
        return self.scan(self._fmt('mrnn_{}'), fn, inputs[0])



class LSTM(Recurrent):
    '''
    '''

    def reset(self, **kwargs):
        '''Reset the weights and biases for this layer to random values.

        Returns
        -------
        count : int
            The number of learnable parameters in this layer.
        '''
        logging.info('initializing lstm layer %s: %s x %s',
                     self.name, self.nin, self.nout)
        self.weights = [
            # these three weight matrices are always diagonal.
            create_vector(self.nout, self._fmt('ci_{}')),
            create_vector(self.nout, self._fmt('cf_{}')),
            create_vector(self.nout, self._fmt('co_{}')),

            create_weights(self.nin, self.nout, self._fmt('xi_{}')),
            create_weights(self.nin, self.nout, self._fmt('xf_{}')),
            create_weights(self.nin, self.nout, self._fmt('xo_{}')),
            create_weights(self.nin, self.nout, self._fmt('xc_{}')),

            create_weights(self.nout, self.nout, self._fmt('hi_{}')),
            create_weights(self.nout, self.nout, self._fmt('hf_{}')),
            create_weights(self.nout, self.nout, self._fmt('ho_{}')),
            create_weights(self.nout, self.nout, self._fmt('hc_{}')),
        ]
        self.biases = [
            create_vector(self.nout, self._fmt('bi_{}')),
            create_vector(self.nout, self._fmt('bf_{}')),
            create_vector(self.nout, self._fmt('bo_{}')),
            create_vector(self.nout, self._fmt('bc_{}')),
        ]
        return self.nout * (7 + 4 * self.nout + 4 * self.nin)

    def transform(self, input):
        '''
        '''
        def fn(x_t, h_tm1, c_tm1,
               W_ci, W_cf, W_co,
               W_xi, W_xf, W_xo, W_xc,
               W_hi, W_hf, W_ho, W_hc,
               b_i, b_f, b_o, b_c):
            i_t = TT.nnet.sigmoid(TT.dot(x_t, W_xi) + TT.dot(h_tm1, W_hi) + c_tm1 * W_ci + b_i)
            f_t = TT.nnet.sigmoid(TT.dot(x_t, W_xf) + TT.dot(h_tm1, W_hf) + c_tm1 * W_cf + b_f)
            c_t = f_t * c_tm1 + i_t * TT.tanh(TT.dot(x_t, W_xc) + TT.dot(h_tm1, W_hc) + b_c)
            o_t = TT.nnet.sigmoid(TT.dot(x_t, W_xo) + TT.dot(h_tm1, W_ho) + c_t * W_co + b_o)
            h_t = o_t * TT.tanh(c_t)
            return h_t, c_t
        (hid, _), updates = theano.scan(
            name=self._fmt('lstm_{}'),
            fn=fn,
            sequences=[input],
            non_sequences=self.weights + self.biases,
            outputs_info=[self.h_0, self.h_0])
        return hid, updates

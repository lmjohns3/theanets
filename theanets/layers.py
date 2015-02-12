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


def random_matrix(nin, nout, mean=0, std=1, sparsity=0, radius=0):
    '''Create a matrix of randomly-initialized weights.

    Parameters
    ----------
    nin : int
        Number of rows of the weight matrix -- equivalently, the number of
        "input" units that the weight matrix connects.
    nout : int
        Number of columns of the weight matrix -- equivalently, the number
        of "output" units that the weight matrix connects.
    mean : float, optional
        Draw initial weight values from a normal with this mean. Defaults to 0.
    std : float, optional
        Draw initial weight values from a normal with this standard deviation.
        Defaults to 1.
    sparsity : float in (0, 1), optional
        If given, ensure that the given fraction of the weight matrix is
        set to zero. Defaults to 0, meaning all weights are nonzero.
    radius : float, optional
        If given, rescale the initial weights to have this spectral radius.
        No scaling is performed by default.

    Returns
    -------
    matrix : numpy array
        An array containing random values. These often represent the weights
        connecting each "input" unit to each "output" unit in a layer.
    '''
    arr = mean + std * np.random.randn(nin, nout)
    if 1 > sparsity > 0:
        k = min(nin, nout)
        mask = np.random.binomial(n=1, p=1 - sparsity, size=(nin, nout)).astype(bool)
        mask[:k, :k] |= np.random.permutation(np.eye(k).astype(bool))
        arr *= mask
    if radius > 0:
        # rescale weights to have the appropriate spectral radius.
        u, s, vT = np.linalg.svd(arr)
        arr = np.dot(np.dot(u, np.diag(radius * s / abs(s[0]))), vT)
    return arr.astype(FLOAT)


def random_vector(size, mean=0, std=1):
    '''Create a vector of randomly-initialized values.

    Parameters
    ----------
    size : int
        Length of vecctor to create.
    mean : float, optional
        Mean value for initial vector values. Defaults to 0.
    std : float, optional
        Standard deviation for initial vector values. Defaults to 1.

    Returns
    -------
    vector : numpy array
        An array containing random values. This often represents the bias for a
        layer of computation units.
    '''
    return (mean + std * np.random.randn(size)).astype(FLOAT)


def softmax(x):
    '''Compute the softmax of the rows of a matrix.

    Parameters
    ----------
    x : theano variable
        A theano matrix. Each row represents one data point, and each column
        represents one of the possible classes for the data points.

    Returns
    -------
    y : theano variable
        A theano expression computing the log-softmax of each row of `x`.
    '''
    z = TT.exp(x - x.max(axis=-1, keepdims=True))
    return z / z.sum(axis=-1, keepdims=True)


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

        # rectification
        'relu': lambda z: TT.maximum(0, z),
        'trel': lambda z: TT.maximum(0, TT.minimum(1, z)),
        'trec': lambda z: TT.maximum(1, z),
        'tlin': lambda z: z * (abs(z) > 1),

        # modifiers
        'rect:max': lambda z: TT.minimum(1, z),
        'rect:min': lambda z: TT.maximum(0, z),

        # normalization
        'norm:dc': lambda z: z - z.mean(axis=-1, keepdims=True),
        'norm:max': lambda z: z / TT.maximum(TT.cast(1e-7, FLOAT), abs(z).max(axis=-1, keepdims=True)),
        'norm:std': lambda z: z / TT.maximum(TT.cast(1e-7, FLOAT), TT.std(z, axis=-1, keepdims=True)),
        'norm:z': lambda z: (z - z.mean(axis=-1, keepdims=True)) / TT.maximum(TT.cast(1e-7, FLOAT), z.std(axis=-1, keepdims=True)),
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
    return input + rng.normal(size=input.shape, std=TT.cast(level, FLOAT), dtype=FLOAT)


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
    return input * rng.binomial(size=input.shape, n=1, p=TT.cast(1, FLOAT)-probability, dtype=FLOAT)


def build(layer, *args, **kwargs):
    '''Construct a layer by name.

    Parameters
    ----------
    layer : str
        The name of the type of layer to build.
    args : tuple
        Positional arguments to pass to the layer constructor.
    kwargs : dict
        Named arguments to pass to the layer constructor.

    Returns
    -------
    layer : :class:`Layer`
        A neural network layer instance.
    '''
    return Layer.build(layer, *args, **kwargs)


class Registrar(type):
    '''A metaclass that builds a registry of its subclasses.'''

    def __init__(cls, name, bases, dct):
        if not hasattr(cls, '_registry'):
            cls._registry = {}
        else:
            cls._registry[name.lower()] = cls
        super(Registrar, cls).__init__(name, bases, dct)

    def build(cls, key, *args, **kwargs):
        return cls._registry[key.lower()](*args, **kwargs)

Base = Registrar(str('Base'), (), {})


def _only(x):
    '''Normalize the type of x to return one element.

    Parameters
    ----------
    x : any
        Either a sequence of elements containing one value, or a non-sequence.

    Raises
    ------
    AssertionError :
        If x is a sequence such that len(x) != 1.

    Returns
    -------
    element : any
        If x is a sequence, returns the first element from the sequence. If x is
        not a sequence, returns x.
    '''
    if hasattr(x, '__len__'):
        assert len(x) == 1
        return x[0]
    return x


class Layer(Base):
    '''Layers in network graphs derive from this base class.

    In ``theanets``, a layer refers to a set of weights and biases, plus the
    "output" units that produce some sort of signal for further layers to
    consume. The first layer in a network, the input layer, is a special case
    with linear activation and no weights or bias.

    Parameters
    ----------
    nin : int or tuple of int
        Size of input vector(s) to this layer.
    nout : int
        Size of output vector produced by this layer.
    name : str, optional
        The name of this layer. If not given, layers will be numbered
        sequentially based on the order in which they are created.
    rng : random number generator, optional
        A theano random number generator to use for creating noise and dropout
        values. If not provided, a new generator will be produced for this
        layer.
    noise : positive float, optional
        Add isotropic gaussian noise with the given standard deviation to the
        output of this layer. Defaults to 0, which does not add any noise to the
        output.
    dropout : float in (0, 1), optional
        Set the given fraction of outputs in this layer randomly to zero.
        Defaults to 0, which does not drop out any units.
    activation : str, optional
        The name of an activation function to use for units in this layer. See
        :func:`build_activation`.
    sparsity : float in (0, 1), optional
        If given, create sparse connections in the layer's weight matrix, such
        that this fraction of the weights is set to zero. By default, this
        parameter is 0, meaning all weights are nonzero.

    Attributes
    ----------
    kwargs : dict
        Keyword arguments that were used when constructing this layer.
    activate : callable
        The activation function to use on this layer's output units.
    params : list of Params
        A list of the parameters in this layer.
    '''

    count = 0

    def __init__(self, **kwargs):
        super(Layer, self).__init__()
        Layer.count += 1
        self.kwargs = kwargs
        self.name = kwargs.get('name', 'layer{}'.format(Layer.count))
        self.nin = kwargs['nin']
        self.nout = kwargs['nout']
        self.activate = create_activation(kwargs.get('activation', 'logistic'))
        self.params = []
        self.setup()

    def output(self, inputs):
        '''Create theano variables representing the output of this layer.

        Parameters
        ----------
        inputs : sequence of theano expressions
            Symbolic inputs to this layer. Usually layers have only one input,
            but layers in general are allowed to have many inputs.

        Returns
        -------
        output : theano expression
            Theano expression specifying the output of this layer.
        monitors : sequence of (name, expression) tuples
            Outputs that can be used to monitor the state of this layer.
        updates : sequence of update tuples
            Updates that should be performed by a theano function that computes
            something using this layer.
        '''
        rng = self.kwargs.get('rng') or RandomStreams()
        noise = self.kwargs.get('noise', 0)
        dropout = self.kwargs.get('dropout', 0)
        out, mon, upd = self.transform(inputs)
        return add_dropout(add_noise(out, noise, rng), dropout, rng), mon, upd

    def transform(self, inputs):
        '''Transform the inputs for this layer into an output for the layer.

        Parameters
        ----------
        inputs : sequence of theano expressions
            Symbolic inputs to this layer. There must be exactly one input.

        Returns
        -------
        output : theano expression
            Theano expression representing the output from this layer.
        monitors : sequence of (name, expression) tuples
            Outputs that can be used to monitor the state of this layer.
        updates : sequence of update tuples
            A sequence of updates to apply inside a theano function.
        '''
        return _only(inputs), (), ()

    def setup(self):
        '''Set up the parameters and initial values for this layer.'''
        pass

    def log_setup(self, count):
        '''Log some information about this layer.

        Parameters
        ----------
        count : int
            Number of parameter values in this layer.
        '''
        logging.info('layer %s: %s -> %s, %s, %d parameters',
                     self.name, self.nin, self.nout,
                     self.activate.__theanets_name__, count)

    def _fmt(self, string):
        '''Helper method to format our name into a string.'''
        if '{' not in string:
            string = '{}_' + string
        return string.format(self.name)

    def _monitors(self, expr, suffix='', levels=None):
        '''Create a list of standard monitor tuples for a given expression.

        Parameters
        ----------
        expr : theano expression
            An expression from the network graph.
        suffix : str, optional
            A suffix to append to monitor names. Defaults to ''.
        levels : sequence of float, optional
            Activation level thresholds for computing monitor expressions.

        Returns
        -------
        monitors : list of (name, expression) tuples
            A list of named monitor expressions.
        '''
        def name(r):
            return '{}{}<{}'.format(self.name, suffix, r)
        def abspct(r):
            return TT.cast(100, FLOAT) * (abs(expr) < TT.cast(r, FLOAT)).mean()
        if levels is None:
            levels = (0.1, 0.2, 0.5, 0.9) if suffix else (0.1, 0.9)
        return [(name(l), abspct(l)) for l in levels]

    def find(self, key):
        '''Get a shared variable for a parameter by name.

        Parameters
        ----------
        key : str or int
            The name of the parameter to look up, or the index of the parameter
            in our parameter list. These are both dependent on the
            implementation of the layer.

        Returns
        -------
        param : shared variable
            A shared variable containing values for the given parameter.

        Raises
        ------
        KeyError
            If a param with the given name does not exist.
        '''
        name = self._fmt(str(key))
        for i, p in enumerate(self.params):
            if key == i or name == p.name:
                return p
        raise KeyError(key)

    def add_weights(self, name, nin=None, nout=None, mean=0, std=None):
        '''Helper method to create a new weight matrix.

        Parameters
        ----------
        name : str
            Name of the parameter to add.
        nin : int, optional
            Size of "input" for this weight matrix. Defaults to self.nin.
        nout : int, optional
            Size of "output" for this weight matrix. Defaults to self.nout.
        mean : float, optional
            Mean value for randomly-initialized weights. Defaults to 0.
        std : float, optional
            Standard deviation of initial matrix values. Defaults to
            :math:`1 / sqrt(n_i + n_o)`.

        Returns
        -------
        count : int
            The number of values in this weight parameter.
        '''
        nin = nin or self.nin
        nout = nout or self.nout
        std = std or 1 / np.sqrt(nin + nout)
        sparsity = self.kwargs.get('sparsity', 0)
        self.params.append(theano.shared(
            random_matrix(nin, nout, mean, std, sparsity=sparsity),
            name=self._fmt(name)))
        return nin * nout

    def add_bias(self, name, nout=None, mean=0, std=1):
        '''Helper method to create a new bias vector.

        Parameters
        ----------
        name : str
            Name of the parameter to add.
        nout : int, optional
            Size of the bias vector. Defaults to self.nout.
        mean : float, optional
            Mean value for randomly-initialized biases. Defaults to 0.
        std : float, optional
            Standard deviation for randomly-initialized biases. Defaults to 1.

        Returns
        -------
        count : int
            The number of values in this bias parameter.
        '''
        nout = nout or self.nout
        self.params.append(theano.shared(
            random_vector(nout, mean, std), name=self._fmt(name)))
        return nout


class Input(Layer):
    '''The input of a network is a special type of layer with no parameters.

    Input layers essentially add only noise to the input data (if desired), but
    otherwise reproduce their inputs exactly.
    '''

    def __init__(self, size, **kwargs):
        kwargs['nin'] = 0
        kwargs['nout'] = size
        kwargs['activation'] = 'linear'
        super(Input, self).__init__(**kwargs)


class Feedforward(Layer):
    '''A feedforward neural network layer performs a transform of its input.

    More precisely, feedforward layers as implemented here perform a weighted
    (affine) transformation of their input, followed by a potentially nonlinear
    "activation" function performed on the transformed input. Feedforward layers
    are the fundamental building block on which most neural network models are
    built.
    '''

    def transform(self, inputs):
        '''Transform the inputs for this layer into an output for the layer.

        Parameters
        ----------
        inputs : sequence of theano expressions
            Symbolic inputs to this layer.

        Returns
        -------
        output : theano expression
            Theano expression representing the output from this layer.
        monitors : sequence of (name, expression) tuples
            Outputs that can be used to monitor the state of this layer.
        updates : sequence of update tuples
            A sequence of updates to apply inside a theano function.
        '''
        if not hasattr(inputs, '__len__'):
            inputs = (inputs, )
        xs = (TT.dot(x, self.find(str(i))) for i, x in enumerate(inputs))
        output = self.activate(sum(xs) + self.find('b'))
        return output, self._monitors(output), ()

    def setup(self):
        '''Set up the parameters and initial values for this layer.'''
        nins = self.nin
        if isinstance(nins, int):
            nins = (nins, )
        count = 0
        for i, nin in enumerate(nins):
            count += self.add_weights(str(i), nin)
        count += self.add_bias('b')
        self.log_setup(count)


class Tied(Feedforward):
    '''A tied-weights feedforward layer shadows weights from another layer.

    Tied weights are typically featured in some types of autoencoder models
    (e.g., PCA). A layer with tied weights requires a "partner" layer -- the
    tied layer borrows the weights from its partner and uses the transpose of
    them to perform its feedforward mapping. Thus, tied layers do not have their
    own weights. On the other hand, tied layers do have their own bias values,
    but these can be fixed to zero during learning to simulate networks with no
    bias (e.g., PCA on mean-centered data).

    Attributes
    ----------
    partner : :class:`Layer`
        The "partner" layer to which this layer is tied.
    '''

    def __init__(self, partner, **kwargs):
        self.partner = partner
        kwargs['nin'] = partner.nout
        kwargs['nout'] = partner.nin
        kwargs['name'] = 'tied-{}'.format(partner.name)
        super(Tied, self).__init__(**kwargs)

    def transform(self, inputs):
        '''Transform the inputs for this layer into an output for the layer.

        Parameters
        ----------
        inputs : sequence of theano expressions
            Symbolic inputs to this layer. There must be exactly one input.

        Returns
        -------
        output : theano expression
            Theano expression representing the output from this layer.
        monitors : sequence of (name, expression) tuples
            Outputs that can be used to monitor the state of this layer.
        updates : sequence of update tuples
            A sequence of updates to apply inside a theano function.
        '''
        preact = TT.dot(_only(inputs), self.partner.find('0').T) + self.find('b')
        output = self.activate(preact)
        return output, self._monitors(output), ()

    def setup(self):
        '''Set up the parameters and initial values for this layer.'''
        # this layer does not create a weight matrix!
        self.log_setup(self.add_bias('b'))


class Classifier(Feedforward):
    '''A classifier layer performs a softmax over a linear input transform.

    Classifier layers are typically the "output" layer of a classifier network.
    '''

    def __init__(self, **kwargs):
        kwargs['activation'] = 'softmax'
        super(Classifier, self).__init__(**kwargs)


class Recurrent(Layer):
    '''A recurrent network layer incorporates some dependency on past values.

    In many respects, a recurrent network layer is much like a basic feedforward
    layer: both layers take an input signal, apply some transformation to it,
    and produce an output signal. Recurrent layers, however, additionally
    preserve the previous state(s) of the layer's output and incorporate them
    into the transformation of the current input.

    This layer type is actually just a base class for the many different types
    of recurrent network layers, for example :class:`RNN` or :class:`LSTM`.

    Parameters
    ----------
    radius : float, optional
        If given, rescale the initial weights for the recurrent units to have
        this spectral radius. No scaling is performed by default.

    direction : {None, 'back', 'backwards'}, optional
        If given, this string indicates whether the recurrency for this layer
        should run "backwards", with future states influencing the current
        state. The default is None, which runs the recurrency forwards in time
        so that past states influence the current state of the layer.
    '''

    def __init__(self, batch_size=64, **kwargs):
        super(Recurrent, self).__init__(**kwargs)

        zeros = np.zeros((batch_size, self.nout), FLOAT)
        self.zeros = lambda s='h': theano.shared(zeros, name=self._fmt('{}0'.format(s)))

    def add_weights(self, name, nin=None, nout=None, mean=0, std=None):
        '''Helper method to create a new weight matrix.

        Parameters
        ----------
        name : str
            Name of parameter to define.
        nin : int, optional
            Size of "input" for this weight matrix. Defaults to self.nin.
        nout : int, optional
            Size of "output" for this weight matrix. Defaults to self.nout.
        mean : float, optional
            Mean of initial matrix values. Defaults to 0.
        std : float, optional
            Standard deviation of initial matrix values. Defaults to
            :math:`1 / sqrt(n_i + n_o)`.

        Returns
        -------
        count : int
            A count of the number of values in this parameter.
        '''
        nin = nin or self.nin
        nout = nout or self.nout
        std = std or 1 / np.sqrt(nin + nout)
        sparsity = self.kwargs.get('sparsity', 0)
        radius = self.kwargs.get('radius', 0) if nin == nout else 0
        self.params.append(theano.shared(
            random_matrix(nin, nout, mean, std, sparsity=sparsity, radius=radius),
            name=self._fmt(name)))
        return nin * nout

    def _scan(self, fn, inputs, inits=None, name='scan'):
        '''Helper method for defining a basic loop in theano.

        Parameters
        ----------
        fn : callable
            The callable to apply in the loop.
        inputs : sequence of theano expressions
            Inputs to the scan operation.
        name : str, optional
            Name of the scan variable to create. Defaults to 'scan'.

        Returns
        -------
        output(s) : theano expression(s)
            Theano expression(s) representing output(s) from the scan.
        updates : sequence of update tuples
            A sequence of updates to apply inside a theano function.
        '''
        return theano.scan(
            fn,
            name=self._fmt(name),
            sequences=inputs,
            outputs_info=inits or [self.zeros()],
            go_backwards='back' in self.kwargs.get('direction', '').lower(),
        )


class RNN(Recurrent):
    '''"Vanilla" recurrent network layer.

    There are many different styles of recurrent network layers, but the one
    implemented here is known as an Elman layer or an SRN (Simple Recurrent
    Network) -- the output from the layer at the previous time step is
    incorporated into the input of the layer at the current time step.
    '''

    def setup(self):
        '''Set up the parameters and initial values for this layer.'''
        self.log_setup(self.add_weights('xh') +
                        self.add_weights('hh', self.nout) +
                        self.add_bias('b'))

    def transform(self, inputs):
        '''Transform the inputs for this layer into an output for the layer.

        Parameters
        ----------
        inputs : sequence of theano expressions
            The inputs to this layer. There must be exactly one input.

        Returns
        -------
        output : theano expression
            Theano expression representing the output from the layer.
        monitors : sequence of (name, expression) tuples
            Outputs that can be used to monitor the state of this layer.
        updates : sequence of update tuples
            A sequence of updates to apply inside a theano function.
        '''
        def fn(x_t, h_tm1):
            return self.activate(x_t + TT.dot(h_tm1, self.find('hh')))
        x = TT.dot(_only(inputs), self.find('xh')) + self.find('b')
        output, updates = self._scan(fn, [x])
        return output, self._monitors(output), updates


class ARRNN(Recurrent):
    '''An adaptive rate RNN defines per-hidden-unit accumulation rates.

    In a normal RNN, a hidden unit is updated completely at each time step,
    :math:`h_t = f(x_t, h_{t-1})`. With an explicit update rate, the state of a
    hidden unit is computed as a mixture of the new and old values, `h_t =
    \alpha_t h_{t-1} + (1 - \alpha_t) f(x_t, h_{t-1})`.

    Rates might be defined in a number of ways: as a vector of values sampled
    randomly from (0, 1), or even as a learnable vector of values. But in the
    adaptive rate RNN, the rate values are computed at each time step as a
    logistic sigmoid applied to an affine transform of the input:
    :math:`\alpha_t = 1 / (1 + e^{-x_t W_{xr} - b_r})`.
    '''

    def setup(self):
        '''Set up the parameters and initial values for this layer.'''
        self.log_setup(self.add_weights('xh') +
                        self.add_weights('xr') +
                        self.add_weights('hh', self.nout) +
                        self.add_bias('b') +
                        self.add_bias('r', std=3))

    def transform(self, inputs):
        '''Transform the inputs for this layer into an output for the layer.

        Parameters
        ----------
        inputs : sequence of theano expressions
            The inputs to this layer. There must be exactly one input.

        Returns
        -------
        output : theano expression
            Theano expression representing the output from the layer.
        monitors : sequence of (name, expression) tuples
            Outputs that can be used to monitor the state of this layer.
        updates : sequence of update tuples
            A sequence of updates to apply inside a theano function.
        '''
        def fn(x_t, r_t, h_tm1):
            h_t = self.activate(x_t + TT.dot(h_tm1, self.find('hh')))
            return r_t * h_tm1 + (1 - r_t) * h_t
        x = _only(inputs)
        h = TT.dot(x, self.find('xh')) + self.find('b')
        r = TT.nnet.sigmoid(TT.dot(x, self.find('xr')) + self.find('r'))
        output, updates = self._scan(fn, [h, r])
        monitors = self._monitors(output) + self._monitors(r, 'rate')
        return output, monitors, updates


class MRNN(Recurrent):
    '''Define a recurrent network layer using multiplicative dynamics.

    The formulation of MRNN implemented here uses a factored dynamics matrix as
    described in Sutskever, Martens & Hinton, ICML 2011, "Generating text with
    recurrent neural networks." This paper is available online at
    http://www.icml-2011.org/papers/524_icmlpaper.pdf.
    '''

    def __init__(self, factors=None, **kwargs):
        self.factors = factors or int(np.ceil(np.sqrt(kwargs['nout'])))
        super(MRNN, self).__init__(**kwargs)

    def setup(self):
        '''Set up the parameters and initial values for this layer.'''
        self.log_setup(
            self.add_weights('xh', self.nin, self.nout) +
            self.add_weights('xf', self.nin, self.factors) +
            self.add_weights('hf', self.nout, self.factors) +
            self.add_weights('fh', self.factors, self.nout) +
            self.add_bias('b'))

    def transform(self, inputs):
        '''Transform the inputs for this layer into an output for the layer.

        Parameters
        ----------
        inputs : sequence of theano expressions
            The inputs to this layer. There must be exactly one input.

        Returns
        -------
        output : theano expression
            Theano expression representing the output from the layer.
        monitors : sequence of (name, expression) tuples
            Outputs that can be used to monitor the state of this layer.
        updates : sequence of update tuples
            A sequence of updates to apply inside a theano function.
        '''
        def fn(x_t, f_t, h_tm1):
            h_t = TT.dot(f_t * TT.dot(h_tm1, self.find('hf')), self.find('fh'))
            return self.activate(x_t + h_t)
        x = _only(inputs)
        h = TT.dot(x, self.find('xh')) + self.find('b')
        f = TT.dot(x, self.find('xf'))
        output, updates = self._scan(fn, [h, f])
        monitors = self._monitors(output) + self._monitors(f, 'fact')
        return output, monitors, updates


class LSTM(Recurrent):
    '''Long Short-Term Memory layer.

    The implementation details for this layer follow the specification given by
    A. Graves, "Generating Sequences with Recurrent Neural Networks,"
    http://arxiv.org/pdf/1308.0850v5.pdf (page 5).
    '''

    def setup(self):
        '''Set up the parameters and initial values for this layer.'''
        self.log_setup(
            self.add_weights('xh', self.nin, 4 * self.nout) +
            self.add_weights('hh', self.nout, 4 * self.nout) +
            self.add_bias('b', 4 * self.nout, mean=3) +
            # the three "peephole" weight matrices are always diagonal.
            self.add_bias('ci', self.nout) +
            self.add_bias('cf', self.nout) +
            self.add_bias('co', self.nout))

    def transform(self, inputs):
        '''Transform the inputs for this layer into an output for the layer.

        Parameters
        ----------
        inputs : sequence of theano expressions
            The inputs to this layer. There must be exactly one input.

        Returns
        -------
        output : theano expression
            Theano expression representing the output from the layer.
        monitors : sequence of (name, expression) tuples
            Outputs that can be used to monitor the state of this layer.
        updates : sequence of update tuples
            A sequence of updates to apply inside a theano function.
        '''
        def split(z):
            n = self.nout
            return z[:, 0*n:1*n], z[:, 1*n:2*n], z[:, 2*n:3*n], z[:, 3*n:4*n]
        def fn(x_t, h_tm1, c_tm1):
            xi, xf, xc, xo = split(x_t + TT.dot(h_tm1, self.find('hh')))
            i_t = TT.nnet.sigmoid(xi + c_tm1 * self.find('ci'))
            f_t = TT.nnet.sigmoid(xf + c_tm1 * self.find('cf'))
            c_t = f_t * c_tm1 + i_t * TT.tanh(xc)
            o_t = TT.nnet.sigmoid(xo + c_t * self.find('co'))
            h_t = o_t * TT.tanh(c_t)
            return h_t, c_t
        x = _only(inputs)
        (output, cell), updates = self._scan(
            fn,
            [TT.dot(x, self.find('xh')) + self.find('b')],
            inits=[self.zeros('h'), self.zeros('c')])
        monitors = self._monitors(output) + self._monitors(cell, 'cell')
        return output, monitors, updates


class Bidirectional(Layer):
    '''A bidirectional recurrent layer runs worker models forward and backward.

    The outputs of the forward and backward passes are combined using an affine
    transformation into the overall output for the layer.

    For an example specification of a bidirectional recurrent network, see A.
    Graves, N. Jaitly, and A. Mohamed, "Hybrid Speech Recognition with Deep
    Bidirectional LSTM," 2013. http://www.cs.toronto.edu/~graves/asru_2013.pdf

    Parameters
    ----------
    worker : str, optional
        This string specifies the type of worker layer to use for the forward
        and backward processing. This parameter defaults to 'rnn' (i.e., vanilla
        recurrent network layer), but can be given as any string that specifies
        a recurrent layer type.
    '''

    def __init__(self, worker='rnn', **kwargs):
        nout = kwargs.pop('nout')
        name = kwargs.pop('name', 'layer{}'.format(Layer.count))
        if 'direction' in kwargs:
            kwargs.pop('direction')
        def make(suffix, direction):
            return build(worker,
                         direction=direction,
                         nout=nout // 2,
                         name='{}_{}'.format(name, suffix),
                         **kwargs)
        self.forward = make('fw', 'forward')
        self.backward = make('bw', 'backward')
        self.params = [self.forward.params, self.backward.params]
        super(Bidirectional, self).__init__(nout=nout, name=name, **kwargs)

    def transform(self, inputs):
        '''Transform the inputs for this layer into an output for the layer.

        Parameters
        ----------
        inputs : sequence of theano expressions
            The inputs to this layer. There must be exactly one input.

        Returns
        -------
        output : theano expression
            Theano expression representing the output from the layer.
        monitors : sequence of (name, expression) tuples
            Outputs that can be used to monitor the state of this layer.
        updates : sequence of update tuples
            A sequence of updates to apply inside a theano function.
        '''
        fx, fm, fu = self.forward.transform(inputs)
        bx, bm, bu = self.backward.transform(inputs)
        return TT.concatenate([fx, bx], axis=2), fm + bm, fu + bu

# -*- coding: utf-8 -*-

r'''This module contains classes for different types of network layers.

.. image:: _static/feedforward_neuron.svg

In a standard feedforward neural network layer, each node :math:`i` in layer
:math:`k` receives inputs from all nodes in layer :math:`k-1`, then transforms
the weighted sum of these inputs:

.. math::
   z_i^k = \sigma\left( b_i^k + \sum_{j=1}^{n_{k-1}} w^k_{ji} z_j^{k-1} \right)

where :math:`\sigma: \mathbb{R} \to \mathbb{R}` is an "activation function" or
"transfer function."  Although many functions will work, typical choices of the
activation function are:

:linear: :math:`\sigma(z) = z`
:rectified linear: :math:`\sigma(z) = \max(0, z)`
:logistic sigmoid: :math:`\sigma(z) = (1 + e^{-z})^{-1}`.

Most activation functions are chosen to incorporate a nonlinearity, since a
model with even multiple linear layers cannot capture nonlinear phenomena. Nodes
in the input layer are assumed to have linear activation (i.e., the input nodes
simply represent the state of the input data), and nodes in the output layer
might have linear or nonlinear activations depending on the modeling task.

Usually all hidden nodes in a network share the same activation function, but
this is not required.
'''

from __future__ import division

import climate
import functools
import numpy as np
import theano
import theano.ifelse
import theano.tensor as TT

from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

logging = climate.get_logger(__name__)

FLOAT = theano.config.floatX


def random_matrix(rows, cols, mean=0, std=1, sparsity=0, radius=0, diagonal=0):
    '''Create a matrix of randomly-initialized weights.

    Parameters
    ----------
    rows : int
        Number of rows of the weight matrix -- equivalently, the number of
        "input" units that the weight matrix connects.
    cols : int
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
    diagonal : float, optional
        If nonzero, create a matrix containing all zeros except for this value
        along the diagonal. If nonzero, other arguments (except for rows and
        cols) will be ignored.

    Returns
    -------
    matrix : numpy array
        An array containing random values. These often represent the weights
        connecting each "input" unit to each "output" unit in a layer.
    '''
    arr = mean + std * np.random.randn(rows, cols)
    if 1 > sparsity > 0:
        k = min(rows, cols)
        mask = np.random.binomial(n=1, p=1 - sparsity, size=(rows, cols)).astype(bool)
        mask[:k, :k] |= np.eye(k).astype(bool)
        arr *= mask
    if radius > 0:
        # rescale weights to have the appropriate spectral radius.
        u, s, vT = np.linalg.svd(arr)
        arr = np.dot(np.dot(u, np.diag(radius * s / abs(s[0]))), vT)
    if diagonal != 0:
        # generate a diagonal weight matrix. ignore other options.
        arr = diagonal * np.eye(max(rows, cols))[:rows, :cols]
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
    x : Theano variable
        A Theano matrix. Each row represents one data point, and each column
        represents one of the possible classes for the data points.

    Returns
    -------
    y : Theano variable
        A Theano expression computing the softmax of each row of `x`.
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


def add_noise(expr, level, rng):
    '''Add noise to elements of the input expression as needed.

    Parameters
    ----------
    expr : Theano expression
        Input expression to add noise to.
    level : float
        Standard deviation of gaussian noise to add to the expression. If this
        is 0, then no gaussian noise is added.

    Returns
    -------
    expr : Theano expression
        The input expression, plus additional noise as specified.
    '''
    if level == 0:
        return expr
    return expr + rng.normal(
        size=expr.shape, std=TT.cast(level, FLOAT), dtype=FLOAT)


def add_dropout(expr, probability, rng):
    '''Add dropouts to elements of the input expression as needed.

    Parameters
    ----------
    expr : Theano expression
        Input expression to add dropouts to.
    probability : float, in [0, 1]
        Probability of dropout for each element of the input. If this is 0,
        then no elements of the input are set randomly to 0.

    Returns
    -------
    expr : Theano expression
        The input expression, plus additional dropouts as specified.
    '''
    if probability == 0:
        return expr
    return expr * rng.binomial(
        size=expr.shape, n=1, p=TT.cast(1, FLOAT)-probability, dtype=FLOAT)


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


class Layer(Base):
    '''Layers in network graphs derive from this base class.

    In ``theanets``, a layer refers to a logically grouped set of parameters and
    computations. Typically this encompasses a set of weight matrix and bias
    vector parameters, plus the "output" units that produce a signal for further
    layers to consume.

    Subclasses of this class can be created to implement many different forms of
    layer-specific computation. For example, a vanilla :class:`Feedforward`
    layer accepts input from the "preceding" layer in a network, computes an
    affine transformation of that input and applies a pointwise transfer
    function. On the other hand, a :class:`Recurrent` layer computes an affine
    transformation of the current input, and combines that with information
    about the state of the layer at previous time steps.

    Most subclasses will need to provide an implementation of the :func:`setup`
    method, which creates the parameters needed by the layer, and the
    :func:`transform` method, which converts the Theano input expressions coming
    in to the layer into some output expression(s).

    Parameters
    ----------
    size : int
        Size of this layer.
    inputs : dict or int
        Size of input(s) to this layer. If one integer is provided, a single
        input of the given size is expected. If a dictionary is provided, it
        maps from output names to corresponding sizes.
    name : str, optional
        The name of this layer. If not given, layers will be numbered
        sequentially based on the order in which they are created.
    activation : str, optional
        The name of an activation function to use for units in this layer. See
        :func:`build_activation`.
    rng : random number generator, optional
        A Theano random number generator to use for creating noise and dropout
        values. If not provided, a new generator will be produced for this
        layer.
    mean, mean_XYZ : float, optional
        Initialize parameters for this layer to have the given mean value. If
        ``mean_XYZ`` is specified, it will apply only to the parameter named
        XYZ. Defaults to 0.
    std, std_XYZ : float, optional
        Initialize parameters for this layer to have the given standard
        deviation. If ``std_XYZ`` is specified, only the parameter named XYZ
        will be so initialized. Defaults to 0.
    sparsity, sparsity_XYZ : float in (0, 1), optional
        If given, create sparse connections in the layer's weight matrix, such
        that this fraction of the weights is set to zero. If ``sparsity_XYZ`` is
        given, it will apply only the parameter with name XYZ. By default, this
        parameter is 0, meaning all weights are nonzero.
    diagonal, diagonal_XYZ : float, optional
        If given, create initial parameter matrices for this layer that are
        initialized to diagonal matrices with this value along the diagonal.
        Defaults to None, which initializes all weights using random values.

    Attributes
    ----------
    name : str
        Name of this layer.
    size : int
        Size of this layer.
    inputs : dict
        Dictionary mapping input names to their corresponding sizes.
    activation : str
        String representing the activation function for this layer.
    kwargs : dict
        Additional keyword arguments used when constructing this layer.
    activate : callable
        The activation function to use on this layer's outputs.
    params : list of Params
        A list of the parameters in this layer.
    '''

    _count = 0

    def __init__(self, size, inputs, name=None, activation='logistic', **kwargs):
        Layer._count += 1
        super(Layer, self).__init__()
        self.size = size
        self.inputs = inputs
        if isinstance(self.inputs, int):
            self.inputs = dict(out=self.inputs)
        self.name = name or '{}{}'.format(
            self.__class__.__name__.lower(), Layer._count)
        self.activation = activation
        self.activate = create_activation(activation)
        self.kwargs = kwargs
        self.params = []
        self.setup()
        self.log()

    @property
    def num_params(self):
        '''Total number of learnable parameters in this layer.'''
        return sum(np.prod(p.get_value().shape) for p in self.params)

    @property
    def output_name(self):
        '''Fully-scoped name of the default output for this layer.'''
        return '{}.out'.format(self.name)

    def connect(self, inputs, noise=0, dropout=0, monitors=None):
        '''Create Theano variables representing the outputs of this layer.

        Parameters
        ----------
        inputs : dict of Theano expressions
            Symbolic inputs to this layer, given as a dictionary mapping string
            names to Theano expressions. Each string key is of the form
            "{layer_name}.{output_name}" (these refer to a specific output from
            a specific layer in the graph) or simply "{output_name}" (these
            typically refer to the outputs from the "most recent" layer; see
            :func:`theanets.graph.Network.setup_layers`).
        noise : positive float or dict, optional
            Add isotropic gaussian noise with the given standard deviation to
            the output of this layer. If this is a scalar, it is applied to the
            default output from this layer; if it is a dictionary, then it
            should map the names of outputs from this layer to the noise to add
            to that layer. Defaults to 0, which does not add noise to any
            outputs.
        dropout : float in (0, 1) or dict, optional
            Set the given fraction of outputs in this layer randomly to zero. If
            this is a scalar, the given value applies to the default output from
            this layer; if it is a dictionary, then it should map the names of
            outputs in this layer to dropout values for that layer. Defaults to
            0, which does not drop out any units in any outputs.
        monitors : dict, optional
            A dictionary mapping string output names to monitors for the given
            output. Typically monitors are computed during training of a network
            to provide insight into the dynamics of the model.

        Returns
        -------
        outputs : dict
            A dictionary mapping names to Theano expressions for the outputs
            from this layer.
        monitors : sequence of (str, expression) tuples
            A sequence of values to compute when "monitoring" the state of this
            layer.
        updates : sequence of (parameter, expression) tuples
            Updates that should be performed by a Theano function that computes
            something using this layer.
        '''
        # prepare monitors/noise/dropouts to be dict types.
        if monitors is None:
            monitors = {}
        if not isinstance(monitors, dict):
            monitors = dict(out=monitors)
        if not isinstance(noise, dict):
            noise = dict(out=noise)
        if not isinstance(dropout, dict):
            dropout = dict(out=dropout)

        # compute output expressions for this layer given the inputs. transform
        # the outputs to be a list of ordered pairs if needed.
        outputs, updates = self.transform(inputs)
        if isinstance(outputs, dict):
            outputs = sorted(outputs.items())
        if isinstance(outputs, TT.TensorVariable):
            outputs = [('out', outputs)]

        # set up outputs, monitors, and updates for this layer.
        rng = self.kwargs.get('rng') or RandomStreams()
        outs = {}
        monits = []
        for name, expr in outputs:
            # add noise and/or dropouts to this output.
            outs[name] = add_dropout(
                add_noise(expr, noise.get(name, 0), rng),
                dropout.get(name, 0), rng)

            # set up monitor expressions for this output.
            levels = monitors.get(name)
            if not levels:
                continue
            if isinstance(levels, dict):
                levels = levels.items()
            for level in levels:
                if isinstance(level, (tuple, list)):
                    label, call = level
                    key = ':{}'.format(label)
                    value = call(expr)
                if isinstance(level, (int, float)):
                    key = '<{}'.format(level)
                    value = (expr < TT.cast(level, FLOAT)).mean()
                monits.append(('{}.{}{}'.format(self.name, name, key), value))

        for param in self.params:
            levels = monitors.get(param.name)
            if not levels:
                continue
            monits.append(('{}.{}{}'.format(self.name, param.name, key), value))

        return outs, monits, updates

    def transform(self, inputs):
        '''Transform the inputs for this layer into an output for the layer.

        Parameters
        ----------
        inputs : dict of Theano expressions
            Symbolic inputs to this layer, given as a dictionary mapping string
            names to Theano expressions. See :func:`Layer.connect`.

        Returns
        -------
        output : Theano expression
            The output for this layer is the same as the input.
        updates : list
            An empty updates list.
        '''
        return inputs['x'], []

    def setup(self):
        '''Set up the parameters and initial values for this layer.'''
        pass

    def log(self):
        '''Log some information about this layer.'''
        act = self.activate.__theanets_name__
        ins = '+'.join('{}:{}'.format(n, s) for n, s in self.inputs.items())
        logging.info('layer %s: %s -> %s, %s, %d parameters',
                     self.name, ins, self.size, act, self.num_params)

    def _fmt(self, string):
        '''Helper method to format our name into a string.'''
        if '{' not in string:
            string = '{}_' + string
        return string.format(self.name)

    def _only_input(self, inputs):
        '''Helper method to retrieve our layer's sole input expression.'''
        assert len(self.inputs) == 1
        return inputs[list(self.inputs)[0]]

    @property
    def input_size(self):
        '''For networks with one input, get the input size.'''
        assert len(self.inputs) == 1
        return list(self.inputs.values())[0]

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

    def add_weights(self, name, nin, nout, mean=0, std=0, sparsity=0, diagonal=0):
        '''Helper method to create a new weight matrix.

        Parameters
        ----------
        name : str
            Name of the parameter to add.
        nin : int
            Size of "input" for this weight matrix.
        nout : int
            Size of "output" for this weight matrix.
        mean : float, optional
            Mean value for randomly-initialized weights. Defaults to 0.
        std : float, optional
            Standard deviation of initial matrix values. Defaults to
            :math:`1 / sqrt(n_i + n_o)`.
        sparsity : float, optional
            Fraction of weights to be set to zero. Defaults to 0.
        diagonal : float, optional
            Initialize weights to a matrix of zeros with this value along the
            diagonal. Defaults to None, which initializes all weights randomly.
        '''
        glorot = 1 / np.sqrt(nin + nout)
        m = self.kwargs.get(
            'mean_{}'.format(name), self.kwargs.get('mean', mean))
        s = self.kwargs.get(
            'std_{}'.format(name), self.kwargs.get('std', std or glorot))
        p = self.kwargs.get(
            'sparsity_{}'.format(name), self.kwargs.get('sparsity', sparsity))
        d = self.kwargs.get(
            'diagonal_{}'.format(name), self.kwargs.get('diagonal', diagonal))
        self.params.append(theano.shared(
            random_matrix(nin, nout, mean=m, std=s, sparsity=p, diagonal=d),
            name=self._fmt(name)))

    def add_bias(self, name, size, mean=0, std=1):
        '''Helper method to create a new bias vector.

        Parameters
        ----------
        name : str
            Name of the parameter to add.
        size : int
            Size of the bias vector.
        mean : float, optional
            Mean value for randomly-initialized biases. Defaults to 0.
        std : float, optional
            Standard deviation for randomly-initialized biases. Defaults to 1.
        '''
        mean = self.kwargs.get('mean_{}'.format(name), mean)
        std = self.kwargs.get('std_{}'.format(name), std)
        self.params.append(theano.shared(
            random_vector(size, mean, std), name=self._fmt(name)))

    def to_spec(self):
        '''Create a specification dictionary for this layer.

        Returns
        -------
        spec : dict
            A dictionary specifying the configuration of this layer.
        '''
        spec = dict(**self.kwargs)
        spec.update(
            form=self.__class__.__name__.lower(),
            name=self.name,
            size=self.size,
            inputs=self.inputs,
            activation=self.activation,
        )
        return spec

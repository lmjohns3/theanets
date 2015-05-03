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

import climate
import functools
import numpy as np
import theano
import theano.tensor as TT
import theano.sparse as SS

from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

logging = climate.get_logger(__name__)

FLOAT = theano.config.floatX


def random_matrix(rows, cols, mean=0, std=1, sparsity=0, radius=0):
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
        A theano expression computing the softmax of each row of `x`.
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
    expr : theano expression
        Input expression to add noise to.
    level : float
        Standard deviation of gaussian noise to add to the expression. If this
        is 0, then no gaussian noise is added.

    Returns
    -------
    expr : theano expression
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
    expr : theano expression
        Input expression to add dropouts to.
    probability : float, in [0, 1]
        Probability of dropout for each element of the input. If this is 0,
        then no elements of the input are set randomly to 0.

    Returns
    -------
    expr : theano expression
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
    :func:`transform` method, which converts the theano input expressions coming
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
        A theano random number generator to use for creating noise and dropout
        values. If not provided, a new generator will be produced for this
        layer.
    sparsity : float in (0, 1), optional
        If given, create sparse connections in the layer's weight matrix, such
        that this fraction of the weights is set to zero. By default, this
        parameter is 0, meaning all weights are nonzero.

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
    num_params : int
        Count of number of parameters in the layer.
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
        self.num_params = 0
        self.setup()

    @property
    def output_name(self):
        '''Fully-scoped name of the default output for this layer.'''
        return '{}.out'.format(self.name)

    def connect(self, inputs, noise=0, dropout=0, monitors=None):
        '''Create theano variables representing the outputs of this layer.

        Parameters
        ----------
        inputs : dict of theano expressions
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
            A dictionary mapping names to theano expressions for the outputs
            from this layer.
        monitors : sequence of (str, expression) tuples
            A sequence of values to compute when "monitoring" the state of this
            layer.
        updates : sequence of (parameter, expression) tuples
            Updates that should be performed by a theano function that computes
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
                    value = TT.cast(100, FLOAT) * (expr < TT.cast(level, FLOAT)).mean()
                monits.append(('{}.{}{}'.format(self.name, name, key), value))

        return outs, monits, updates

    def transform(self, inputs):
        '''Transform the inputs for this layer into an output for the layer.

        Parameters
        ----------
        inputs : dict of theano expressions
            Symbolic inputs to this layer, given as a dictionary mapping string
            names to Theano expressions. See :func:`Layer.connect`.

        Returns
        -------
        output : theano expression
            The output for this layer is the same as the input.
        updates : list
            An empty updates list.
        '''
        return inputs['x'], []

    def setup(self):
        '''Set up the parameters and initial values for this layer.'''
        pass

    def log_setup(self):
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

    def add_weights(self, name, nin, nout, mean=0, std=None, sparsity=0):
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
        '''
        mean = self.kwargs.get(
            'mean_{}'.format(name), self.kwargs.get('mean', mean))
        std = self.kwargs.get(
            'std_{}'.format(name), self.kwargs.get(
                'std', std or 1 / np.sqrt(nin + nout)))
        sparsity = self.kwargs.get(
            'sparsity_{}'.format(name),
            self.kwargs.get('sparsity', sparsity))
        self.params.append(theano.shared(
            random_matrix(nin, nout, mean, std, sparsity=sparsity),
            name=self._fmt(name)))
        self.num_params += nin * nout

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
        mean = self.kwargs.get('mean_{}'.format(name), self.kwargs.get('mean', mean))
        std = self.kwargs.get('std_{}'.format(name), self.kwargs.get('std', std))
        self.params.append(theano.shared(
            random_vector(size, mean, std), name=self._fmt(name)))
        self.num_params += size

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


class Input(Layer):
    '''The input of a network is a special type of layer with no parameters.

    Input layers essentially add only noise to the input data (if desired), but
    otherwise reproduce their inputs exactly.
    '''

    def __init__(self, **kwargs):
        kwargs['inputs'] = 0
        kwargs['activation'] = 'linear'
        super(Input, self).__init__(**kwargs)

    def to_spec(self):
        '''Create a specification for this layer.

        Returns
        -------
        spec : int
            A single integer specifying the size of this layer.
        '''
        return self.size


class Feedforward(Layer):
    '''A feedforward neural network layer performs a transform of its input.

    More precisely, feedforward layers as implemented here perform an affine
    transformation of their input, followed by a potentially nonlinear
    "activation" function performed elementwise on the transformed input.

    Feedforward layers are the fundamental building block on which most neural
    network models are built.
    '''

    def transform(self, inputs):
        '''Transform the inputs for this layer into an output for the layer.

        Parameters
        ----------
        inputs : dict of theano expressions
            Symbolic inputs to this layer, given as a dictionary mapping string
            names to Theano expressions. See :func:`Layer.connect`.

        Returns
        -------
        outputs : dict of theano expressions
            A map from string output names to Theano expressions for the outputs
            from this layer. This layer type generates a "pre" output that gives
            the unit activity before applying the layer's activation function,
            and an "out" output that gives the post-activation output.
        updates : list of update pairs
            An empty list of updates to apply from this layer.
        '''
        def _dot(x, y):
            if isinstance(x, SS.SparseVariable):
                return SS.structured_dot(x, y)
            else:
                return TT.dot(x, y)
        def weight(n):
            return 'w' if len(self.inputs) == 1 else 'w_{}'.format(n)
        xws = ((inputs[n], self.find(weight(n))) for n in self.inputs)
        pre = sum(_dot(x, w) for x, w in xws) + self.find('b')
        return dict(pre=pre, out=self.activate(pre)), []

    def setup(self):
        '''Set up the parameters and initial values for this layer.'''
        for name, size in self.inputs.items():
            label = 'w' if len(self.inputs) == 1 else 'w_{}'.format(name)
            self.add_weights(label, size, self.size)
        self.add_bias('b', self.size)
        self.log_setup()


class Classifier(Feedforward):
    '''A classifier layer performs a softmax over a linear input transform.

    Classifier layers are typically the "output" layer of a classifier network.

    This layer type really only wraps the output activation of a standard
    :class:`Feedforward` layer.
    '''

    def __init__(self, **kwargs):
        kwargs['activation'] = 'softmax'
        super(Classifier, self).__init__(**kwargs)


class Tied(Layer):
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
        kwargs['inputs'] = partner.size
        kwargs['size'] = list(partner.inputs.values())[0]
        kwargs['name'] = 'tied-{}'.format(partner.name)
        super(Tied, self).__init__(**kwargs)

    def transform(self, inputs):
        '''Transform the inputs for this layer into an output for the layer.

        Parameters
        ----------
        inputs : dict of theano expressions
            Symbolic inputs to this layer, given as a dictionary mapping string
            names to Theano expressions. See :func:`Layer.connect`.

        Returns
        -------
        outputs : dict of theano expressions
            A map from string output names to Theano expressions for the outputs
            from this layer. This layer type generates a "pre" output that gives
            the unit activity before applying the layer's activation function,
            and an "out" output that gives the post-activation output.
        updates : list of update pairs
            An empty sequence of updates.
        '''
        x = inputs[list(self.inputs)[0]]
        pre = TT.dot(x, self.partner.find('w').T) + self.find('b')
        return dict(pre=pre, out=self.activate(pre)), []

    def setup(self):
        '''Set up the parameters and initial values for this layer.'''
        # this layer does not create a weight matrix!
        self.add_bias('b', self.size)
        self.log_setup()

    def to_spec(self):
        '''Create a specification dictionary for this layer.

        Returns
        -------
        spec : dict
            A dictionary specifying the configuration of this layer.
        '''
        spec = super(Tied, self).to_spec()
        spec['partner'] = self.partner.name
        return spec


class Maxout(Layer):
    '''A maxout layer computes a piecewise linear activation function.

    '''

    def __init__(self, **kwargs):
        self.pieces = kwargs.pop('pieces')
        super(Maxout, self).__init__(**kwargs)

    def setup(self):
        '''Set up the parameters and initial values for this layer.'''
        self.add_weights('w')
        self.add_bias('b', self.size)
        logging.info('layer %s: %s -> %s (x%s), %s, %d parameters',
                     self.name,
                     list(self.inputs.values())[0],
                     self.size,
                     self.pieces,
                     self.activate.__theanets_name__,
                     self.num_params)

    def transform(self, inputs):
        '''Transform the inputs for this layer into an output for the layer.

        Parameters
        ----------
        inputs : dict of theano expressions
            Symbolic inputs to this layer, given as a dictionary mapping string
            names to Theano expressions. See :func:`Layer.connect`.

        Returns
        -------
        outputs : dict of theano expressions
            A map from string output names to Theano expressions for the outputs
            from this layer. This layer type generates a "pre" output that gives
            the unit activity before applying the layer's activation function,
            and an "out" output that gives the post-activation output.
        updates : list of update pairs
            An empty sequence of state updates.
        '''
        x = inputs[list(self.inputs)[0]]
        pre = TT.dot(x, self.find('w')).max(axis=2) + self.find('b')
        return dict(pre=pre, out=self.activate(pre)), []

    def add_weights(self, name, mean=0, std=None, sparsity=0):
        '''Helper method to create a new weight matrix.

        Parameters
        ----------
        name : str
            Name of the parameter to add.
        mean : float, optional
            Mean value for randomly-initialized weights. Defaults to 0.
        std : float, optional
            Standard deviation of initial matrix values. Defaults to
            :math:`1 / sqrt(n_i + n_o)`.
        sparsity : float, optional
            Fraction of weights to set to zero. Defaults to 0.
        '''
        nin = list(self.inputs.values())[0]
        def rm():
            return random_matrix(
                nin, self.size, mean, std or 1 / np.sqrt(nin + self.size),
                sparsity=self.kwargs.get('sparsity', sparsity),
            )[:, :, None]
        # stack up weight matrices for the pieces in our maxout.
        arr = np.concatenate([rm() for _ in range(self.pieces)], axis=2)
        self.params.append(theano.shared(arr, name=self._fmt(name)))
        self.num_params += nin * self.size * self.pieces

    def to_spec(self):
        '''Create a specification dictionary for this layer.

        Returns
        -------
        spec : dict
            A dictionary specifying the configuration of this layer.
        '''
        spec = super(Maxout, self).to_spec()
        spec['pieces'] = self.pieces
        return spec


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

    def __init__(self, **kwargs):
        super(Recurrent, self).__init__(**kwargs)

    def initial_state(self, name, batch_size):
        '''Return an array of suitable for representing initial state.

        Parameters
        ----------
        name : str
            Name of the variable to return.
        batch_size : int
            Number of elements in a batch. This can be symbolic.

        Returns
        -------
        initial : theano shared variable
            A variable containing the initial state of some recurrent variable.
        '''
        values = theano.shared(
            np.zeros((1, self.size), FLOAT),
            name=self._fmt('{}0'.format(name)))
        return TT.repeat(values, batch_size, axis=0)

    def add_weights(self, name, nin, nout, mean=0, std=None, sparsity=0, radius=0):
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
        sparsity : float, optional
            Fraction of weights to set randomly to zero. Defaults to 0.
        radius : float, optional
            If nonzero, rescale initial weights to have this spectral radius.
            Defaults to 0.
        '''
        std = std or 1 / np.sqrt(nin + nout)
        sparsity = self.kwargs.get('sparsity', sparsity)
        radius = self.kwargs.get('radius', radius) if nin == nout else 0
        self.params.append(theano.shared(
            random_matrix(nin, nout, mean, std, sparsity=sparsity, radius=radius),
            name=self._fmt(name)))
        self.num_params += nin * nout

    def _scan(self, fn, inputs, inits=None, name='scan'):
        '''Helper method for defining a basic loop in theano.

        Parameters
        ----------
        fn : callable
            The callable to apply in the loop.
        inputs : sequence of theano expressions
            Inputs to the scan operation.
        inits : sequence of None, tensor, tuple, or scan output specifier
            Specifiers for the outputs of the scan operation. This should be a
            list containing:
            - None for values that are output by the scan but not tapped as
              inputs,
            - a theano tensor variable with a 'shape' attribute, or
            - a tuple containing a string and an integer for output values that
              are also tapped as inputs, or
            - a dictionary containing a full output specifier.
            See "outputs_info" in the Theano documentation for ``scan``.
        name : str, optional
            Name of the scan variable to create. Defaults to 'scan'.

        Returns
        -------
        output(s) : theano expression(s)
            Theano expression(s) representing output(s) from the scan.
        updates : sequence of update tuples
            A sequence of updates to apply inside a theano function.
        '''
        outputs = []
        for i, x in enumerate(inits or inputs):
            if hasattr(x, 'shape'):
                x = self.initial_state(str(i), x.shape[1])
            elif isinstance(x, int):
                x = self.initial_state(str(i), x)
            elif isinstance(x, tuple):
                x = self.initial_state(*x)
            outputs.append(x)
        return theano.scan(
            fn,
            name=self._fmt(name),
            sequences=inputs,
            outputs_info=outputs,
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
        self.add_weights('xh', list(self.inputs.values())[0], self.size)
        self.add_weights('hh', self.size, self.size)
        self.add_bias('b', self.size)
        self.log_setup()

    def transform(self, inputs):
        '''Transform the inputs for this layer into an output for the layer.

        Parameters
        ----------
        inputs : dict of theano expressions
            Symbolic inputs to this layer, given as a dictionary mapping string
            names to Theano expressions. See :func:`Layer.connect`.

        Returns
        -------
        outputs : dict of theano expressions
            A map from string output names to Theano expressions for the outputs
            from this layer. This layer type generates a "pre" output that gives
            the unit activity before applying the layer's activation function,
            and an "out" output that gives the post-activation output.
        updates : list of update pairs
            A sequence of updates to apply inside a theano function.
        '''
        def fn(x_t, h_tm1):
            pre = x_t + TT.dot(h_tm1, self.find('hh'))
            return [pre, self.activate(pre)]
        x = TT.dot(inputs[list(self.inputs)[0]], self.find('xh')) + self.find('b')
        (pre, out), updates = self._scan(fn, [x], [None, x])
        return dict(pre=pre, out=out), updates


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
        self.add_weights('xh', list(self.inputs.values())[0], self.size)
        self.add_weights('hh', self.size, self.size)
        self.add_bias('b', self.size)
        self.add_bias('r', self.size, mean=2, std=1)
        self.log_setup()

    def transform(self, inputs):
        '''Transform the inputs for this layer into an output for the layer.

        Parameters
        ----------
        inputs : dict of theano expressions
            Symbolic inputs to this layer, given as a dictionary mapping string
            names to Theano expressions. See :func:`Layer.connect`.

        Returns
        -------
        outputs : theano expression
            A map from string output names to Theano expressions for the outputs
            from this layer. This layer type generates a "pre" output that gives
            the unit activity before applying the layer's activation function,
            a "hid" output that gives the rate-independent, post-activation
            hidden state, a "rate" output that gives the rate value for each
            hidden unit, and an "out" output that gives the hidden output.
        updates : list of update pairs
            A sequence of updates to apply inside a theano function.
        '''
        r = TT.nnet.sigmoid(self.find('r'))
        x = inputs[list(self.inputs)[0]]
        h = TT.dot(x, self.find('xh')) + self.find('b')
        def fn(x_t, h_tm1):
            pre = x_t + TT.dot(h_tm1, self.find('hh'))
            h_t = self.activate(pre)
            return [pre, h_t, r * h_tm1 + (1 - r) * h_t]
        (pre, hid, out), updates = self._scan(fn, [h], [None, None, x])
        return dict(pre=pre, hid=hid, rate=r, out=out), updates


class MRNN(Recurrent):
    '''Define a recurrent network layer using multiplicative dynamics.

    The formulation of MRNN implemented here uses a factored dynamics matrix as
    described in Sutskever, Martens & Hinton, ICML 2011, "Generating text with
    recurrent neural networks." This paper is available online at
    http://www.icml-2011.org/papers/524_icmlpaper.pdf.
    '''

    def __init__(self, factors=None, **kwargs):
        self.factors = factors or int(np.ceil(np.sqrt(kwargs['size'])))
        super(MRNN, self).__init__(**kwargs)

    def setup(self):
        '''Set up the parameters and initial values for this layer.'''
        nin = list(self.inputs.values())[0]
        self.add_weights('xh', nin, self.size)
        self.add_weights('xf', nin, self.factors)
        self.add_weights('hf', self.size, self.factors)
        self.add_weights('fh', self.factors, self.size)
        self.add_bias('b', self.size)
        self.log_setup()

    def transform(self, inputs):
        '''Transform the inputs for this layer into an output for the layer.

        Parameters
        ----------
        inputs : dict of theano expressions
            Symbolic inputs to this layer, given as a dictionary mapping string
            names to Theano expressions. See :func:`Layer.connect`.

        Returns
        -------
        outputs : dict of theano expressions
            A map from string output names to Theano expressions for the outputs
            from this layer. This layer type generates a "factors" output that
            gives the activation of the hidden weight factors given the input
            data (but not incorporating influence from the hidden states), a
            "pre" output that gives the unit activity before applying the
            layer's activation function, and an "out" output that gives the
            post-activation output.
        updates : list of update pairs
            A sequence of updates to apply inside a theano function.
        '''
        def fn(x_t, f_t, h_tm1):
            pre = x_t + TT.dot(f_t * TT.dot(h_tm1, self.find('hf')), self.find('fh'))
            return [pre, self.activate(pre)]
        x = inputs[list(self.inputs)[0]]
        h = TT.dot(x, self.find('xh')) + self.find('b')
        f = TT.dot(x, self.find('xf'))
        (pre, out), updates = self._scan(fn, [h, f], [None, x])
        return dict(pre=pre, factors=f, out=out), updates

    def to_spec(self):
        '''Create a specification dictionary for this layer.

        Returns
        -------
        spec : dict
            A dictionary specifying the configuration of this layer.
        '''
        spec = super(MRNN, self).to_spec()
        spec['factors'] = self.factors
        return spec


class LSTM(Recurrent):
    '''Long Short-Term Memory layer.

    The implementation details for this layer follow the specification given by
    A. Graves, "Generating Sequences with Recurrent Neural Networks,"
    http://arxiv.org/pdf/1308.0850v5.pdf (page 5).
    '''

    def setup(self):
        '''Set up the parameters and initial values for this layer.'''
        self.add_weights('xh', list(self.inputs.values())[0], 4 * self.size)
        self.add_weights('hh', self.size, 4 * self.size)
        self.add_bias('b', 4 * self.size, mean=2)
        # the three "peephole" weight matrices are always diagonal.
        self.add_bias('ci', self.size)
        self.add_bias('cf', self.size)
        self.add_bias('co', self.size)
        self.log_setup()

    def transform(self, inputs):
        '''Transform the inputs for this layer into an output for the layer.

        Parameters
        ----------
        inputs : dict of theano expressions
            Symbolic inputs to this layer, given as a dictionary mapping string
            names to Theano expressions. See :func:`Layer.connect`.

        Returns
        -------
        outputs : dict of theano expressions
            A map from string output names to Theano expressions for the outputs
            from this layer. This layer type generates a "cell" output that
            gives the value of each hidden cell in the layer, and an "out"
            output that gives the actual gated output from the layer.
        updates : list of update pairs
            A sequence of updates to apply inside a theano function.
        '''
        def split(z):
            n = self.size
            return z[:, 0*n:1*n], z[:, 1*n:2*n], z[:, 2*n:3*n], z[:, 3*n:4*n]
        def fn(x_t, h_tm1, c_tm1):
            xi, xf, xc, xo = split(x_t + TT.dot(h_tm1, self.find('hh')))
            i_t = TT.nnet.sigmoid(xi + c_tm1 * self.find('ci'))
            f_t = TT.nnet.sigmoid(xf + c_tm1 * self.find('cf'))
            c_t = f_t * c_tm1 + i_t * TT.tanh(xc)
            o_t = TT.nnet.sigmoid(xo + c_t * self.find('co'))
            h_t = o_t * TT.tanh(c_t)
            return [h_t, c_t]
        x = inputs[list(self.inputs)[0]]
        batch_size = x.shape[1]
        (out, cell), updates = self._scan(
            fn,
            [TT.dot(x, self.find('xh')) + self.find('b')],
            [('h', batch_size), ('c', batch_size)])
        return dict(out=out, cell=cell), updates


class GRU(Recurrent):
    '''Gated Recurrent Unit layer.

    The implementation is from J Chung, C Gulcehre, KH Cho, & Y Bengio (2014),
    "Empirical Evaluation of Gated Recurrent Neural Networks on Sequence
    Modeling" (page 4), available at http://arxiv.org/abs/1412.3555v1.
    '''
    def setup(self):
        self.add_weights('xh', list(self.inputs.values())[0], 3 * self.size)
        self.add_weights('hh', self.size, 3 * self.size)
        self.add_bias('b', 3 * self.size)

    def transform(self, inputs):
        '''Transform inputs to this layer into outputs for the layer.

        Parameters
        ----------
        inputs : dict of theano expressions
            Symbolic inputs to this layer, given as a dictionary mapping string
            names to Theano expressions. See :func:`Layer.connect`.

        Returns
        -------
        outputs : dict of theano expressions
            A map from string output names to Theano expressions for the outputs
            from this layer. This layer type generates a "pre" output that gives
            the unit activity before applying the layer's activation function, a
            "hid" output that gives the post-activation values before applying
            the rate mixing, and an "out" output that gives the overall output.
        updates : sequence of update pairs
            A sequence of updates to apply to this layer's state inside a theano
            function.
        '''
        def split(z):
            n = self.size
            return z[:, 0*n:1*n], z[:, 1*n:2*n], z[:, 2*n:3*n]
        def fn(x_t, h_tm1):
            xh, xz, xr = split(x_t + TT.dot(h_tm1, self.find('hh')))
            z = TT.nnet.sigmoid(xz)
            pre = xh + TT.nnet.sigmoid(xr) * h_tm1
            h_t = self.activate(pre)
            return [pre, h_t, z, (1 - z) * h_tm1 + z * h_t]
        x = TT.dot(inputs[list(self.inputs)[0]], self.find('xh')) + self.find('b')
        (pre, hid, rate, out), updates = self._scan(fn, [x], [None, None, None, x])
        return dict(pre=pre, hid=hid, rate=rate, out=out), updates


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
        size = kwargs.pop('size')
        name = kwargs.pop('name', 'layer{}'.format(Layer.count))
        if 'direction' in kwargs:
            kwargs.pop('direction')
        def make(suffix, direction):
            return build(worker,
                         direction=direction,
                         size=size // 2,
                         name='{}_{}'.format(name, suffix),
                         **kwargs)
        self.forward = make('fw', 'forward')
        self.backward = make('bw', 'backward')
        self.params = [self.forward.params, self.backward.params]
        super(Bidirectional, self).__init__(size=size, name=name, **kwargs)

    def transform(self, inputs):
        '''Transform the inputs for this layer into an output for the layer.

        Parameters
        ----------
        inputs : dict of theano expressions
            Symbolic inputs to this layer, given as a dictionary mapping string
            names to Theano expressions. See :func:`Layer.connect`.

        Returns
        -------
        outputs : dict of theano expressions
            Theano expression representing the output from the layer. This layer
            type produces "pre" and "hid" outputs that concatenates the outputs
            from its underlying workers. It also passes along the individual
            outputs from its workers using "fw_" and "bw_" prefixes for forward
            and backward directions.
        updates : list of update pairs
            A list of state updates to apply inside a theano function.
        '''
        fout, fupd = self.forward.transform(inputs)
        bout, bupd = self.backward.transform(inputs)
        out = TT.concatenate([fout['out'], bout['out']], axis=2)
        pre = TT.concatenate([fout['pre'], bout['pre']], axis=2)
        outputs = dict(out=out, pre=pre)
        for k, v in fout.items():
            outputs['fw_{}'.format(k)] = v
        for k, v in bout.items():
            outputs['bw_{}'.format(k)] = v
        return outputs, fupd + bupd

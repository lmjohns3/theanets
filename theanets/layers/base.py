# -*- coding: utf-8 -*-

r'''This module contains classes for different types of network layers.

.. image:: _static/feedforward_neuron.svg

In a standard feedforward neural network layer, each node :math:`i` in layer
:math:`k` receives inputs from all nodes in layer :math:`k-1`, then transforms
the weighted sum of these inputs:

.. math::
   z_i^k = \sigma\left( b_i^k + \sum_{j=1}^{n_{k-1}} w^k_{ji} z_j^{k-1} \right)

where :math:`\sigma: \mathbb{R} \to \mathbb{R}` is an :mod:`activation function
<theanets.activations>`.

In addition to standard feedforward layers, other types of layers are also
commonly used:

- For recurrent models, :mod:`recurrent layers <theanets.layers.recurrent>`
  permit a cycle in the computation graph that depends on a previous time step.

- For models that process images, :mod:`convolution layers
  <theanets.layers.convolution>` are common.

- For some types of autoencoder models, it is common to :class:`tie layer weights to
  another layer <theanets.layers.feedforward.Tied>`.
'''

from __future__ import division

import climate
import numpy as np
import theano
import theano.sparse as SS
import theano.tensor as TT

from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

from .. import activations
from .. import util

logging = climate.get_logger(__name__)

FLOAT = theano.config.floatX


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
    return expr + rng.normal(size=expr.shape, std=level, dtype=FLOAT)


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
        size=expr.shape, n=1, p=1-probability, dtype=FLOAT)


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


class Layer(util.Registrar(str('Base'), (), {})):
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
    nrng : :class:`numpy.random.RandomState` or int, optional
        A numpy random number generator, or an integer seed for a random number
        generator. If not provided, the random number generator will be created
        with an automatically chosen seed.
    trng : Theano random number generator, optional
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
    activate : callable
        The activation function to use on this layer's outputs.
    kwargs : dict
        Additional keyword arguments used when constructing this layer.
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
        self.nrng = kwargs.get('nrng')
        if self.nrng is None or isinstance(self.nrng, int):
            self.nrng = np.random.RandomState(self.nrng)
        self.activation = activation
        self.activate = activations.build(activation, self)
        self.kwargs = kwargs
        self._params = []
        self.setup()
        self.log()

    @property
    def params(self):
        '''A list of all parameters in this layer.'''
        return self._params + getattr(self.activate, 'params', [])

    @property
    def num_params(self):
        '''Total number of learnable parameters in this layer.'''
        return sum(np.prod(p.get_value().shape) for p in self.params)

    def output_name(self, name='out'):
        '''Return a fully-scoped name for the given layer output.

        Parameters
        ----------
        name : str
            Name of an output for this layer.

        Returns
        -------
        scoped : str
            A fully-scoped name for the given output from this layer.
        '''
        return '{}:{}'.format(self.name, name)

    def connect(self, inputs, noise=0, dropout=0):
        '''Create Theano variables representing the outputs of this layer.

        Parameters
        ----------
        inputs : dict of Theano expressions
            Symbolic inputs to this layer, given as a dictionary mapping string
            names to Theano expressions. Each string key should be of the form
            "{layer_name}:{output_name}" and refers to a specific output from
            a specific layer in the graph.
        noise : dict of noise values, optional
            This dictionary should map the names of outputs from this layer to
            the zero-mean, isotropic noise to add to that output. Defaults to 0,
            which does not add noise to any outputs.
        dropout : dict of dropout values, optional
            This dictionary should map the names of outputs in this layer to
            dropout values for that output. Defaults to 0, which does not drop
            out any units in any outputs.

        Returns
        -------
        outputs : dict
            A dictionary mapping names to Theano expressions for the outputs
            from this layer.
        updates : sequence of (parameter, expression) tuples
            Updates that should be performed by a Theano function that computes
            something using this layer.
        '''
        outputs, updates = self.transform(inputs)

        # transform the outputs to be a list of ordered pairs if needed.
        if isinstance(outputs, dict):
            outputs = sorted(outputs.items())
        if isinstance(outputs, (TT.TensorVariable, SS.SparseVariable)):
            outputs = [('out', outputs)]

        # set up outputs for this layer by adding noise and dropout as needed.
        rng = self.kwargs.get('trng') or RandomStreams()
        if not isinstance(noise, dict):
            noise = {self.output_name(): noise}
        if not isinstance(dropout, dict):
            dropout = {self.output_name(): dropout}
        outs = {}
        for name, expr in outputs:
            scoped = self.output_name(name)
            noisy = add_noise(expr, noise.get(scoped, 0), rng)
            dropped = add_dropout(noisy, dropout.get(scoped, 0), rng)
            outs[scoped] = dropped

        return outs, updates

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
        inputs = ', '.join('({}){}'.format(n, s) for n, s in self.inputs.items())
        logging.info('layer %s "%s": %s -> %s, %s, %d parameters',
                     self.__class__.__name__,
                     self.name,
                     inputs,
                     self.size,
                     getattr(self.activate, 'name', self.activate),
                     self.num_params)

    def _fmt(self, string):
        '''Helper method to format our name into a string.'''
        if '{' not in string:
            string = '{}.' + string
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
        for i, p in enumerate(self._params):
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
        self._params.append(theano.shared(
            util.random_matrix(nin, nout, mean=m, std=s, sparsity=p,
                               diagonal=d, rng=self.nrng),
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
        self._params.append(theano.shared(
            util.random_vector(size, mean, std, rng=self.nrng),
            name=self._fmt(name)))

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

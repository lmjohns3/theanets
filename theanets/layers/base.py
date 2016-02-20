# -*- coding: utf-8 -*-

'''This module contains classes for different types of network layers.'''

from __future__ import division

import climate
import numpy as np
import theano
import theano.sparse as SS
import theano.tensor as TT

from .. import activations
from .. import util

logging = climate.get_logger(__name__)

__all__ = [
    'Concatenate',
    'Input',
    'Flatten',
    'Layer',
    'Product',
    'Reshape',
]


class Layer(util.Registrar(str('Base'), (), {})):
    '''Base class for network layers.

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
    inputs : str or tuple of str, optional
        Name(s) of input(s) to this layer. These names must be
        :func:`resolved <resolve>` to layers by :func:`binding <bind>` the layer
        inside a :class:`network graph <theanets.graph.Network>`. Defaults to an
        empty tuple; in practice this needs to be provided for most layers.
    name : str, optional
        The name of this layer. If not given, layers will be numbered
        sequentially based on the order in which they are created.
    activation : str, optional
        The name of an activation function to use for units in this layer. See
        :func:`build_activation`.
    rng : :class:`numpy.random.RandomState` or int, optional
        A numpy random number generator, or an integer seed for a random number
        generator. If not provided, the random number generator will be created
        with an automatically chosen seed.
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
    inputs : tuple of str
        Name(s) of input(s) to this layer.
    activate : callable
        The activation function to use on this layer's outputs.
    kwargs : dict
        Additional keyword arguments used when constructing this layer.
    '''

    _count = 0

    def __init__(self, size, inputs=(), name=None, **kwargs):
        super(Layer, self).__init__()

        self.size = size
        self.kwargs = kwargs
        self._params = []

        if isinstance(inputs, (tuple, list)):
            self.inputs = tuple(inputs)
        else:
            self.inputs = (inputs, )
        self._resolved_inputs = {}

        Layer._count += 1
        self.name = name or '{}{}'.format(
            self.__class__.__name__.lower(), Layer._count)

        self.rng = kwargs.get('rng', kwargs.get('nrng'))
        if self.rng is None or isinstance(self.rng, int):
            self.rng = np.random.RandomState(self.rng)

        self.activate = activations.build(kwargs.get('activation', 'relu'), self)

    @property
    def params(self):
        '''A list of all parameters in this layer.'''
        return self._params + getattr(self.activate, 'params', [])

    @property
    def output_name(self):
        '''Full name of the default output for this layer.'''
        return self.full_name('out')

    @property
    def input_size(self):
        '''For networks with one input, get the input size.'''
        assert len(self.inputs) == 1
        return self._resolved_inputs[self.inputs[0]].size

    def full_name(self, name):
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

    def connect(self, inputs):
        '''Create Theano variables representing the outputs of this layer.

        Parameters
        ----------
        inputs : dict of Theano expressions
            Symbolic inputs to this layer, given as a dictionary mapping string
            names to Theano expressions. Each string key should be of the form
            "{layer_name}:{output_name}" and refers to a specific output from
            a specific layer in the graph.

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
        outs = {self.full_name(name): expr for name, expr in outputs}
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
        raise NotImplementedError

    def bind(self, graph, reset=True, initialize=True):
        '''Bind this layer into a computation graph.

        This method is a wrapper for performing common initialization tasks. It
        calls :func:`resolve`, :func:`setup`, and :func:`log`.

        Parameters
        ----------
        graph : :class:`Network <theanets.graph.Network>`
            A computation network in which this layer is to be bound.
        reset : bool, optional
            If ``True`` (the default), reset the resolved layers for this layer.
        initialize : bool, optional
            If ``True`` (the default), initialize the parameters for this layer
            by calling :func:`setup`.

        Raises
        ------
        theanets.util.ConfigurationError :
            If an input cannot be resolved.
        '''
        if reset:
            self._resolved_inputs = {}
        self.resolve(graph.layers)
        if initialize:
            self.setup()
        self.log()

    def resolve(self, layers):
        '''Resolve the names of inputs for this layer into layer objects.

        Parameters
        ----------
        layers : list of :class:`Layer`
            A list of the layers that are available for resolving inputs.

        Raises
        ------
        theanets.util.ConfigurationError :
            If an input cannot be resolved.
        '''
        keys = []
        for name in self.inputs:
            matches = [l for l in layers if name.split(':')[0] == l.name]
            if len(matches) != 1:
                raise util.ConfigurationError(
                    'layer "{}" cannot resolve input "{}" using {}'
                    .format(self.name, name, [l.name for l in layers]))
            full = name if ':' in name else matches[0].output_name
            self._resolved_inputs[full] = matches[0]
            keys.append(full)
        self.inputs = tuple(keys)

    def setup(self):
        '''Set up the parameters and initial values for this layer.'''
        pass

    def log(self):
        '''Log some information about this layer.'''
        inputs = ', '.join('({0}){1.size}'.format(n, l)
                           for n, l in self._resolved_inputs.items())
        logging.info('layer %s "%s": %s -> %s, %s, %d parameters',
                     self.__class__.__name__,
                     self.name,
                     inputs,
                     self.size,
                     getattr(self.activate, 'name', self.activate),
                     sum(np.prod(p.get_value().shape) for p in self.params))

    def _fmt(self, string):
        '''Helper method to format our name into a string.'''
        if '{' not in string:
            string = '{}.' + string
        return string.format(self.name)

    def _only_input(self, inputs):
        '''Helper method to retrieve our layer's sole input expression.'''
        assert len(self.inputs) == 1
        return inputs[self.inputs[0]]

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
                               diagonal=d, rng=self.rng),
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
            util.random_vector(size, mean, std, rng=self.rng),
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
            activation=self.kwargs.get('activation', 'relu'),
        )
        return spec


class Input(Layer):
    '''A layer that receives external input data.

    Input layers are responsible for the Theano variables that represent input
    to a network. The name of the layer is passed along to the symbolic Theano
    input variable.

    Input layers essentially add only noise to the input data (if desired), but
    otherwise reproduce their inputs exactly.

    Parameters
    ----------
    ndim : int, optional
        Number of dimensions required to store the input data for this layer.
        Defaults to 2: ``(num-examples, num-variables)``.
    sparse : bool or str, optional
        If this is ``'csr'`` or ``'csc'``, then the inputs to the loss will be
        stored as sparse matrices in the CSR or CSC format (respectively). If
        this is True, sparse input will be enabled in CSR format. By default
        this is False, which means inputs are dense.

    Raises
    ------
    AssertionError :
        If ``sparse`` is enabled and ``ndim`` is not 2.
    '''

    def __init__(self, size, name='in', ndim=2, sparse=False, **kwargs):
        self.input = util.FLOAT_CONTAINERS[ndim](name)
        if sparse is True or \
           isinstance(sparse, util.basestring) and sparse.lower() == 'csr':
            assert ndim == 2, 'Theano only supports sparse arrays with 2 dims'
            self.input = SS.csr_matrix('input')
        if isinstance(sparse, util.basestring) and sparse.lower() == 'csc':
            assert ndim == 2, 'Theano only supports sparse arrays with 2 dims'
            self.input = SS.csc_matrix('input')
        super(Input, self).__init__(
            size=size, name=name, activation='linear', ndim=ndim, sparse=sparse)

    def log(self):
        '''Log some information about this layer.'''
        logging.info('layer %s "%s": %s input channel%s',
                     self.__class__.__name__, self.name, self.size,
                     's' if self.size > 1 else '')

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
        return self.input, []


class Product(Layer):
    '''Multiply several inputs together elementwise.

    Notes
    -----

    This layer performs an elementwise multiplication of multiple inputs; all
    inputs must be the same shape.

    *Outputs*

    - ``out`` --- elementwise product of its inputs
    '''

    __extra_registration_keys__ = ['prod']

    def transform(self, inputs):
        '''Transform the inputs for this layer into an output for the layer.

        Parameters
        ----------
        inputs : dict of Theano expressions
            Symbolic inputs to this layer, given as a dictionary mapping string
            names to Theano expressions. See :func:`Layer.connect`.

        Returns
        -------
        outputs : dict of Theano expressions
            A map from string output names to Theano expressions for the outputs
            from this layer. This layer type generates a "pre" output that gives
            the unit activity before applying the layer's activation function,
            and an "out" output that gives the post-activation output.
        updates : list of update pairs
            An empty sequence of updates.
        '''
        return dict(out=np.prod([inputs[k] for k in self.inputs])), []


class Flatten(Layer):
    '''Flatten all but the batch index of the input.

    Notes
    -----

    In ``theanets``, the leading axis of a data array always runs over the
    examples in a mini-batch. Since the number of examples in a mini-batch is
    constant throughout a network graph, this layer always preserves the shape
    of the leading axis of its inputs.

    This layer type flattens all of the non-leading dimensions of its inputs
    into one dimension. If you'd like to perform an arbitrary reshape of the
    input data, use a :class:`Reshape` layer.

    *Outputs*

    - ``out`` --- flattened inputs
    '''

    __extra_registration_keys__ = ['flat']

    def transform(self, inputs):
        '''Transform the inputs for this layer into an output for the layer.

        Parameters
        ----------
        inputs : dict of Theano expressions
            Symbolic inputs to this layer, given as a dictionary mapping string
            names to Theano expressions. See :func:`Layer.connect`.

        Returns
        -------
        outputs : dict of Theano expressions
            A map from string output names to Theano expressions for the outputs
            from this layer. This layer type generates a "pre" output that gives
            the unit activity before applying the layer's activation function,
            and an "out" output that gives the post-activation output.
        updates : list of update pairs
            An empty sequence of updates.
        '''
        x = self._only_input(inputs)
        return dict(out=x.reshape([x.shape[0], -1])), []


class Concatenate(Layer):
    '''Concatenate multiple inputs along the last axis.

    Notes
    -----

    This layer concatenates multiple inputs along their last dimension; all
    inputs must have the same dimensionality and the same shape along all but
    the last dimension. The size of this layer must equal the sum of the sizes
    of the inputs.

    *Outputs*

    - ``out`` --- inputs concatenated along last axis
    '''

    __extra_registration_keys__ = ['concat']

    def transform(self, inputs):
        '''Transform the inputs for this layer into an output for the layer.

        Parameters
        ----------
        inputs : dict of Theano expressions
            Symbolic inputs to this layer, given as a dictionary mapping string
            names to Theano expressions. See :func:`Layer.connect`.

        Returns
        -------
        outputs : dict of Theano expressions
            A map from string output names to Theano expressions for the outputs
            from this layer. This layer type generates a "pre" output that gives
            the unit activity before applying the layer's activation function,
            and an "out" output that gives the post-activation output.
        updates : list of update pairs
            An empty sequence of updates.
        '''
        # using axis=-1 doesn't work with concatenate!
        tensors = [inputs[k] for k in self.inputs]
        out = TT.concatenate(tensors, axis=tensors[0].ndim - 1)
        return dict(out=out), []


class Reshape(Layer):
    '''Reshape an input to have different numbers of dimensions.

    Notes
    -----

    In ``theanets``, the leading axis of a data array always runs over the
    examples in a mini-batch. Since the number of examples in a mini-batch is
    constant throughout a network graph, this layer always preserves the shape
    of the leading axis of its inputs.

    If you want to vectorize a data array, you could do that using ``(-1, )`` as
    the shape for this layer. But it's often easier to read if you use the
    :class:`Flatten` layer type to reshape a layer's output into a flat vector.

    *Outputs*

    - ``out`` --- reshaped inputs

    Parameters
    ----------
    shape : sequence of int
        The desired shape of the output "vectors" for this layer. This should
        not include the leading axis of the actual shape of the data arrays
        processed by the graph! For example, to reshape input vectors of length
        a * b into 2D output "images" use ``(a, b)`` as the shape---not
        ``(batch-size, a, b)``.

    Attributes
    ----------
    shape : list of int
        The desired shape of the output "vectors" for this layer.
    '''

    def __init__(self, shape, **kwargs):
        self.shape = list(shape)
        kwargs['size'] = shape[-1]
        super(Reshape, self).__init__(**kwargs)
        assert self.size == self.shape[-1]

    def transform(self, inputs):
        '''Transform the inputs for this layer into an output for the layer.

        Parameters
        ----------
        inputs : dict of Theano expressions
            Symbolic inputs to this layer, given as a dictionary mapping string
            names to Theano expressions. See :func:`Layer.connect`.

        Returns
        -------
        outputs : dict of Theano expressions
            A map from string output names to Theano expressions for the outputs
            from this layer. This layer type generates a "pre" output that gives
            the unit activity before applying the layer's activation function,
            and an "out" output that gives the post-activation output.
        updates : list of update pairs
            An empty sequence of updates.
        '''
        x = self._only_input(inputs)
        return dict(out=x.reshape([x.shape[0]] + self.shape)), []

    def to_spec(self):
        '''Create a specification dictionary for this layer.

        Returns
        -------
        spec : dict
            A dictionary specifying the configuration of this layer.
        '''
        spec = super(Reshape, self).to_spec()
        spec['shape'] = self.shape
        return spec

# -*- coding: utf-8 -*-

r'''Utility functions and classes.'''

import fnmatch
import numpy as np
import theano

FLOAT = theano.config.floatX


class Registrar(type):
    '''A metaclass that builds a registry of its subclasses.'''

    def __init__(cls, name, bases, dct):
        if not hasattr(cls, '_registry'):
            cls._registry = {}
        else:
            cls._registry[name.lower()] = cls
            for name in getattr(cls, '__extra_registration_keys__', ()):
                cls._registry[name.lower()] = cls
        super(Registrar, cls).__init__(name, bases, dct)

    def build(cls, key, *args, **kwargs):
        return cls._registry[key.lower()](*args, **kwargs)

    def get_class(cls, key):
        return cls._registry[key.lower()]

    def is_registered(cls, key):
        return key.lower() in cls._registry


def random_matrix(rows, cols, mean=0, std=1, sparsity=0, radius=0, diagonal=0, rng=None):
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
    rng : :class:`numpy.random.RandomState` or int, optional
        A random number generator, or an integer seed for a random number
        generator. If not provided, the random number generator will be created
        with an automatically chosen seed.

    Returns
    -------
    matrix : numpy array
        An array containing random values. These often represent the weights
        connecting each "input" unit to each "output" unit in a layer.
    '''
    if rng is None or isinstance(rng, int):
        rng = np.random.RandomState(rng)
    arr = mean + std * rng.randn(rows, cols)
    if 1 > sparsity > 0:
        k = min(rows, cols)
        mask = rng.binomial(n=1, p=1 - sparsity, size=(rows, cols)).astype(bool)
        mask[:k, :k] |= np.eye(k).astype(bool)
        arr *= mask
    if radius > 0:
        # rescale weights to have the appropriate spectral radius.
        u, s, vT = np.linalg.svd(arr, full_matrices=False)
        arr = np.dot(np.dot(u, np.diag(radius * s / abs(s[0]))), vT)
    if diagonal != 0:
        # generate a diagonal weight matrix. ignore other options.
        arr = diagonal * np.eye(max(rows, cols))[:rows, :cols]
    return arr.astype(FLOAT)


def random_vector(size, mean=0, std=1, rng=None):
    '''Create a vector of randomly-initialized values.

    Parameters
    ----------
    size : int
        Length of vecctor to create.
    mean : float, optional
        Mean value for initial vector values. Defaults to 0.
    std : float, optional
        Standard deviation for initial vector values. Defaults to 1.
    rng : :class:`numpy.random.RandomState` or int, optional
        A random number generator, or an integer seed for a random number
        generator. If not provided, the random number generator will be created
        with an automatically chosen seed.

    Returns
    -------
    vector : numpy array
        An array containing random values. This often represents the bias for a
        layer of computation units.
    '''
    if rng is None or isinstance(rng, int):
        rng = np.random.RandomState(rng)
    return (mean + std * rng.randn(size)).astype(FLOAT)


def outputs_matching(outputs, pattern):
    '''Get the outputs from a network that match a pattern.

    Parameters
    ----------
    outputs : dict or sequence of (str, theano expression)
        Output expressions to filter for matches. If this is a dictionary, its
        ``items()`` will be processed for matches.
    pattern : str
        A glob-style pattern to match against. ``'*'`` will match all outputs,
        while ``'a:*'`` matches outputs named ``'a:b'`` ``'a:xyz'`` and so on.

    Yields
    ------
    matches : pair of str, theano expression
        Generates a sequence of (name, expression) pairs. The name is the name
        of the output that matched, and the expression is the symbolic output in
        the network graph.
    '''
    if isinstance(outputs, dict):
        outputs = outputs.items()
    for name, expr in outputs:
        if fnmatch.fnmatch(name, pattern):
            yield name, expr


def params_matching(network, pattern):
    '''Get the parameters from a network that match a pattern.

    Parameters
    ----------
    network : :class:`theanets.graph.Network`
        A network to retrieve parameters from.
    pattern : str
        A glob-style pattern to match against. ``'*'`` will match all
        parameters, while ``'a.*'`` matches parameters named ``'a.b'``
        ``'a.xyz'`` and so on.

    Yields
    ------
    matches : pair of str, theano expression
        Generates a sequence of (name, expression) pairs. The name is the name
        of the parameter that matched, and the expression represents the
        parameter symbolically.
    '''
    for p in network.params:
        if fnmatch.fnmatch(p.name, pattern):
            yield p.name, p

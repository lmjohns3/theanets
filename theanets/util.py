# -*- coding: utf-8 -*-

r'''Utility functions and classes.'''

import numpy as np
import theano


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

    def is_registered(cls, key):
        return key.lower() in cls._registry


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

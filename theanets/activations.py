# -*- coding: utf-8 -*-

r'''Activation functions for network layers.
'''

import functools
import numpy as np
import theano
import theano.tensor as TT

from . import util

FLOAT = theano.config.floatX


def build(name, layer, **kwargs):
    '''Construct an activation function by name.

    Parameters
    ----------
    name : str or :class:`Activation`
        The name of the type of activation function to build, or an
        already-created instance of an activation function.
    layer : :class:`theanets.layers.Layer`
        The layer to which this activation will be applied.
    kwargs : dict
        Additional named arguments to pass to the activation constructor.

    Returns
    -------
    activation : :class:`Activation`
        A neural network activation function instance.
    '''
    if isinstance(name, Activation):
        return name
    def compose(a, b):
        c = lambda z: b(a(z))
        c.name = ['%s(%s)' % (b.name, a.name)]
        return c
    if '+' in name:
        return functools.reduce(compose, (build(n) for n in name.split('+')))
    act = {
        # s-shaped
        'tanh':        TT.tanh,
        'logistic':    TT.nnet.sigmoid,
        'sigmoid':     TT.nnet.sigmoid,

        # softmax (typically for classification)
        'softmax':     softmax,

        # linear variants
        'linear':      lambda x: x,
        'softplus':    TT.nnet.softplus,
        'relu':        lambda x: (x + abs(x)) / 2,
        'rect:max':    lambda x: (1 + x - abs(x - 1)) / 2,
        'rect:minmax': lambda x: (1 + abs(x) - abs(x - 1)) / 2,

        # batch normalization
        'norm:mean':   lambda x: x - x.mean(axis=-1, keepdims=True),
        'norm:max':    lambda x: x / (
            abs(x).max(axis=-1, keepdims=True) + TT.cast(1e-6, FLOAT)),
        'norm:std':    lambda x: x / (
            x.std(axis=-1, keepdims=True) + TT.cast(1e-6, FLOAT)),
        'norm:z':      lambda x: (x - x.mean(axis=-1, keepdims=True)) / (
            x.std(axis=-1, keepdims=True) + TT.cast(1e-6, FLOAT)),
    }.get(name)
    if act is not None:
        act.name = name
        act.params = []
        return act
    return Activation.build(name, name, layer, **kwargs)


def softmax(x):
    z = TT.exp(x - x.max(axis=-1, keepdims=True))
    return z / z.sum(axis=-1, keepdims=True)


class Activation(util.Registrar(str('Base'), (), {})):
    '''An activation function for a neural network layer.

    Parameters
    ----------
    name : str
        Name of this activation function.
    layer : :class:`Layer`
        The layer to which this function is applied.

    Attributes
    ----------
    name : str
        Name of this activation function.
    layer : :class:`Layer`
        The layer to which this function is applied.
    '''

    def __init__(self, name, layer, **kwargs):
        self.name = name
        self.layer = layer
        self.kwargs = kwargs
        self.params = []

    def __call__(self, x):
        '''Compute a symbolic expression for this activation function.

        Parameters
        ----------
        x : Theano expression
            A Theano expression representing the input to this activation
            function.

        Returns
        -------
        y : Theano expression
            A Theano expression representing the output from this activation
            function.
        '''
        raise NotImplementedError


class Prelu(Activation):
    __extra_registration_keys__ = ['leaky-relu']

    def __init__(self, *args, **kwargs):
        super(Prelu, self).__init__(*args, **kwargs)
        self.leak = theano.shared(
            np.ones((self.layer.size, ), FLOAT) * 0.1,
            name=self.layer._fmt('leak'))
        self.params.append(self.leak)

    def __call__(self, x):
        return (x + abs(x)) / 2 + self.leak * (x - abs(x)) / 2


class LGrelu(Activation):
    __extra_registration_keys__ = ['leaky-gain-relu']

    def __init__(self, *args, **kwargs):
        super(LGrelu, self).__init__(*args, **kwargs)
        self.gain = theano.shared(
            np.ones((self.layer.size, ), FLOAT),
            name=self.layer._fmt('gain'))
        self.params.append(self.gain)
        self.leak = theano.shared(
            np.ones((self.layer.size, ), FLOAT) * 0.1,
            name=self.layer._fmt('leak'))
        self.params.append(self.leak)

    def __call__(self, x):
        return self.gain * (x + abs(x)) / 2 + self.leak * (x - abs(x)) / 2


class Maxout(Activation):
    r'''The maxout activation function is piecewise linear.

    This activation is unusual in that it requires a parameter: the number of
    pieces to use. Given :math:`k` pieces, this activation computes a slope and
    an intercept for each linear piece. It then transfers the input activation
    as the maximum of all of the pieces.

    Parameters
    ----------
    pieces : int
        Number of linear pieces to use in the activation.
    '''

    def __init__(self, *args, **kwargs):
        super(Maxout, self).__init__(*args, **kwargs)

        self.pieces = kwargs['pieces']

        m = np.ones((self.layer.size, self.pieces), FLOAT)
        self.slope = theano.shared(m, name=self.layer._fmt('slope'))
        self.params.append(self.slope)

        b = np.ones((self.pieces, ), FLOAT)
        self.intercept = theano.shared(v, name=self.layer._fmt('intercept'))
        self.params.append(self.intercept)

    def __call__(self, x):
        return (x * self.slope + self.intercept).max(axis=-1)

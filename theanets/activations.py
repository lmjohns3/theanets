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
    return Activation.build(name, name, layer, **kwargs)


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


class Linear(Activation):
    def __call__(self, x):
        return x


class Tanh(Activation):
    def __call__(self, x):
        return TT.tanh(x)

class Logistic(Activation):
    __extra_registration_keys__ = ['sigmoid']
    def __call__(self, x):
        return TT.nnet.sigmoid(x)


class Softmax(Activation):
    def __call__(self, x):
        z = TT.exp(x - x.max(axis=-1, keepdims=True))
        return z / z.sum(axis=-1, keepdims=True)


class Softplus(Activation):
    def __call__(self, x):
        return TT.nnet.softplus(x)

class Relu(Activation):
    __extra_registration_keys__ = ['rect:min']
    def __call__(self, x):
        return (x + abs(x)) / 2

class RectMax(Activation):
    __extra_registration_keys__ = ['rect:max']
    def __call__(self, x):
        return (1 + x - abs(x - 1)) / 2

class TruncatedRelu(Activation):
    __extra_registration_keys__ = ['trelu', 'rect:minmax']
    def __call__(self, x):
        return (1 + abs(x) - abs(x - 1)) / 2

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


class NormMean(Activation):
    __extra_registration_keys__ = ['norm:mean']
    def __call__(self, x):
        return x - x.mean(axis=-1, keepdims=True)

class NormMax(Activation):
    __extra_registration_keys__ = ['norm:max']
    def __call__(self, x):
        s = abs(x).max(axis=-1, keepdims=True)
        return x / (s + TT.cast(1e-6, FLOAT))

class NormStd(Activation):
    __extra_registration_keys__ = ['norm:std']
    def __call__(self, x):
        s = x.std(axis=-1, keepdims=True)
        return x / (s + TT.cast(1e-6, FLOAT))

class NormZ(Activation):
    __extra_registration_keys__ = ['norm:z']
    def __call__(self, x):
        c = (x - x.mean(axis=-1, keepdims=True))
        s = x.std(axis=-1, keepdims=True)
        return c / (s + TT.cast(1e-6, FLOAT))

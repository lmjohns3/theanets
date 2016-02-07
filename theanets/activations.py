# -*- coding: utf-8 -*-

r'''Activation functions for network layers.

Activation functions are normally constructed using the :func:`build` function.
Commonly available functions are:

- "linear"
- "logistic" (or "sigmoid")
- "tanh"
- "softmax" (typically used for :class:`classifier <theanets.feedforward.Classifier>`
  output layers)
- "relu" (or "rect:max")
- "rect:min"
- "rect:minmax"
- "softplus" (continuous approximation of "relu")
- "norm:mean": subtractive (mean) batch normalization
- "norm:max": divisive (max) batch normalization
- "norm:std": divisive (standard deviation) batch normalization
- "norm:z": z-score batch normalization

Additionally, the names of all classes defined in this module can be used as
keys when building an activation function.
'''

import functools
import numpy as np
import theano
import theano.tensor as TT

from . import util


def _identity(x): return x


def _relu(x): return (x + abs(x)) / 2


def _trel(x): return (x + 1 - abs(x - 1)) / 2


def _rect(x): return (abs(x) + 1 - abs(x - 1)) / 2


def _norm_mean(x): return x - x.mean(axis=-1, keepdims=True)


def _norm_max(x): return x / (abs(x).max(axis=-1, keepdims=True) + 1e-8)


def _norm_std(x): return x / (x.std(axis=-1, keepdims=True) + 1e-8)


def _norm_z(x): return ((x - x.mean(axis=-1, keepdims=True)) /
                        (x.std(axis=-1, keepdims=True) + 1e-8))


def _softmax(x):
    z = TT.exp(x - x.max(axis=-1, keepdims=True))
    return z / z.sum(axis=-1, keepdims=True)


COMMON = {
    # s-shaped
    'tanh':        TT.tanh,
    'logistic':    TT.nnet.sigmoid,
    'sigmoid':     TT.nnet.sigmoid,

    # softmax (typically for classification)
    'softmax':     _softmax,

    # linear variants
    'linear':      _identity,
    'softplus':    TT.nnet.softplus,
    'relu':        _relu,
    'rect:max':    _relu,
    'rect:min':    _trel,
    'rect:minmax': _rect,

    # batch normalization
    'norm:mean':   _norm_mean,
    'norm:max':    _norm_max,
    'norm:std':    _norm_std,
    'norm:z':      _norm_z,
}


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
        def c(z): return b(a(z))
        c.name = '%s(%s)' % (b.name, a.name)
        c.params = getattr(b, 'params', []) + getattr(a, 'params', [])
        return c

    if '+' in name:
        return functools.reduce(
            compose, (build(n, layer, **kwargs) for n in name.split('+')))

    act = COMMON.get(name)
    if act is not None:
        act.name = name
        act.params = []
        return act

    if name.lower().startswith('maxout') and ':' in name:
        name, pieces = name.split(':', 1)
        kwargs['pieces'] = int(pieces)
    kwargs['name'] = name
    kwargs['layer'] = layer
    return Activation.build(name, **kwargs)


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
    r'''Parametric rectified linear activation with learnable leak rate.

    This activation is characterized by two linear pieces joined at the origin.
    For negative inputs, the unit response is a linear function of the input
    with slope :math:`r` (the "leak rate"). For positive inputs, the unit
    response is the identity function:

    .. math::
       f(x) = \left\{ \begin{eqnarray*} rx &\qquad& \mbox{if } x < 0 \\
                      x &\qquad& \mbox{otherwise} \end{eqnarray*} \right.

    This activation allocates a separate leak rate for each unit in its layer.

    References
    ----------
    K He, X Zhang, S Ren, J Sun (2015), "Delving Deep into Rectifiers:
    Surpassing Human-Level Performance on ImageNet Classification"
    http://arxiv.org/abs/1502.01852
    '''

    __extra_registration_keys__ = ['leaky-relu']

    def __init__(self, *args, **kwargs):
        super(Prelu, self).__init__(*args, **kwargs)
        self.leak = theano.shared(
            0.1 * abs(self.layer.rng.randn(self.layer.size).astype(util.FLOAT)),
            name=self.layer._fmt('leak'))
        self.params.append(self.leak)

    def __call__(self, x):
        return (x + abs(x)) / 2 + self.leak * (x - abs(x)) / 2


class LGrelu(Activation):
    r'''Rectified linear activation with learnable leak rate and gain.

    This activation is characterized by two linear pieces joined at the origin.
    For negative inputs, the unit response is a linear function of the input
    with slope :math:`r` (the "leak rate"). For positive inputs, the unit
    response is a different linear function of the input with slope :math:`g`
    (the "gain"):

    .. math::
       f(x) = \left\{ \begin{eqnarray*} rx &\qquad& \mbox{if } x < 0 \\
                       gx &\qquad& \mbox{otherwise} \end{eqnarray*} \right.

    This activation allocates a separate leak and gain rate for each unit in its
    layer.
    '''

    __extra_registration_keys__ = ['leaky-gain-relu']

    def __init__(self, *args, **kwargs):
        super(LGrelu, self).__init__(*args, **kwargs)
        self.gain = theano.shared(
            0.1 * abs(self.layer.rng.randn(self.layer.size).astype(util.FLOAT)),
            name=self.layer._fmt('gain'))
        self.params.append(self.gain)
        self.leak = theano.shared(
            0.1 * abs(self.layer.rng.randn(self.layer.size).astype(util.FLOAT)),
            name=self.layer._fmt('leak'))
        self.params.append(self.leak)

    def __call__(self, x):
        return self.gain * (x + abs(x)) / 2 + self.leak * (x - abs(x)) / 2


class Maxout(Activation):
    r'''Arbitrary piecewise linear activation.

    This activation is unusual in that it requires a parameter at initialization
    time: the number of linear pieces to use. Consider a layer for the moment
    with just one unit. A maxout activation with :math:`k` pieces uses a slope
    :math:`m_k` and an intercept :math:`b_k` for each linear piece. It then
    transforms the input to the maximum of all of the pieces:

    .. math::
       f(x) = \max_k m_k x + b_k

    The parameters :math:`m_k` and :math:`b_k` are learnable.

    For layers with more than one unit, the maxout activation allocates a slope
    :math:`m_{ki}` and intercept :math:`b_{ki}` for each unit :math:`i` and each
    piece :math:`k`. The activation for unit :math:`x_i` is:

    .. math::
       f(x_i) = \max_k m_{ki} x_i + b_{ki}

    Again, the slope and intercept parameters are learnable.

    This activation is actually a generalization of the rectified linear
    activations; to see how, just allocate 2 pieces and set the intercepts to 0.
    The slopes of the ``relu`` activation are given by :math:`m = (0, 1)`, those
    of the :class:`Prelu` function are given by :math:`m = (r, 1)`, and those of
    the :class:`LGrelu` are given by :math:`m = (r, g)` where :math:`r` is the
    leak rate parameter and :math:`g` is a gain parameter.

    .. note::

       To use this activation in a network layer specification, provide an
       activation string of the form ``'maxout:k'``, where ``k`` is an integer
       giving the number of piecewise functions.

       For example, the layer tuple ``(100, 'rnn', 'maxout:10')`` specifies a
       vanilla :class:`RNN <theanets.layers.recurrent.RNN>` layer with 100 units
       and a maxout activation with 10 pieces.

    Parameters
    ----------
    pieces : int
        Number of linear pieces to use in the activation.
    '''

    def __init__(self, *args, **kwargs):
        super(Maxout, self).__init__(*args, **kwargs)

        self.pieces = kwargs['pieces']

        m = self.layer.rng.randn(self.layer.size, self.pieces).astype(util.FLOAT)
        self.slope = theano.shared(m, name=self.layer._fmt('slope'))
        self.params.append(self.slope)

        b = self.layer.rng.randn(self.layer.size, self.pieces).astype(util.FLOAT)
        self.intercept = theano.shared(b, name=self.layer._fmt('intercept'))
        self.params.append(self.intercept)

    def __call__(self, x):
        dims = list(range(x.ndim)) + ['x']
        return (x.dimshuffle(*dims) * self.slope + self.intercept).max(axis=-1)

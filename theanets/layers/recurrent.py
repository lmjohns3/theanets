# -*- coding: utf-8 -*-

'''Recurrent layers allow time dependencies in the computation graph.'''

from __future__ import division

import climate
import numpy as np
import theano
import theano.tensor as TT

from . import base
from .. import util

logging = climate.get_logger(__name__)

__all__ = [
    'Bidirectional',
    'Clockwork',
    'GRU',
    'LSTM',
    'MRNN',
    'MUT1',
    'RNN',
    'RRNN',
    'SCRN',
]


class Recurrent(base.Layer):
    r'''A recurrent network layer incorporates some dependency on past values.

    In many respects, a recurrent network layer is much like a basic feedforward
    layer: both layers take an input signal, apply some transformation to it,
    and produce an output signal. Recurrent layers, however, additionally
    preserve the previous state(s) of the layer's output and incorporate them
    into the transformation of the current input.

    This layer type is actually just a base class for the many different types
    of recurrent network layers, for example :class:`RNN` or :class:`LSTM`.

    Recurrent layer types can only be included in ``theanets`` models that use
    recurrent inputs and outputs, i.e., :class:`theanets.recurrent.Autoencoder`,
    :class:`theanets.recurrent.Predictor`,
    :class:`theanets.recurrent.Classifier`, or
    :class:`theanets.recurrent.Regressor`.

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

    bptt_limit : int, optional
        If given, limit backpropagation of gradient information in scans (loops)
        to the given number of time steps. Defaults to -1, which imposes no
        limit.

    h_0 : str, optional
        If provided, this should name a network output that provides the initial
        state for the network's hidden units. Defaults to ``None``, which uses
        an all-zero initial state.
    '''

    def __init__(self, h_0=None, **kwargs):
        super(Recurrent, self).__init__(**kwargs)
        self.h_0 = h_0

    def resolve(self, layers):
        '''Resolve the names of inputs for this layer.'''
        super(Recurrent, self).resolve(layers)
        self.h_0 = self._resolve(self.h_0, layers)

    def add_weights(self, name, nin, nout, mean=0, std=0, sparsity=0, radius=0,
                    diagonal=0):
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
        glorot = 1 / np.sqrt(nin + nout)
        mean = self.kwargs.get(
            'mean_{}'.format(name), self.kwargs.get('mean', mean))
        std = self.kwargs.get(
            'std_{}'.format(name), self.kwargs.get('std', std or glorot))
        s = self.kwargs.get(
            'sparsity_{}'.format(name), self.kwargs.get('sparsity', sparsity))
        r = self.kwargs.get(
            'radius_{}'.format(name), self.kwargs.get('radius', radius))
        d = self.kwargs.get(
            'diagonal_{}'.format(name), self.kwargs.get('diagonal', diagonal))
        if nin == self.size and nout % nin == 0:
            arr = np.concatenate([
                util.random_matrix(nin, nin, mean, std, sparsity=s, radius=r,
                                   diagonal=d, rng=self.rng)
                for _ in range(nout // nin)], axis=1)
        else:
            arr = util.random_matrix(nin, nout, mean, std, sparsity=s, rng=self.rng)
        self._params.append(theano.shared(arr, name=self._fmt(name)))

    def _resolve(self, value, layers):
        '''Helper for resolving the name of one input to a full name.

        Parameters
        ----------
        value : str or None
            Name of the attribute to resolve
        layers : list of :class:`Layer <theanets.layers.base.Layer>`
            A list of the layers that are available for resolving inputs.

        Raises
        ------
        theanets.util.ConfigurationError :
            If an input cannot be resolved.

        Returns
        -------
        name : str or None
            A fully-scoped input name, or ``None`` if ``value`` was ``None``.
        '''
        if value is None or ':' in value:
            return value
        try:
            layer = [l for l in layers if l.name == value][0]
        except:
            raise util.ConfigurationError(
                'layer "{}" cannot resolve input "{}"'.format(self.name, value))
        return layer.output_name

    def _scan(self, inputs, outputs, name='scan', step=None, constants=None):
        '''Helper method for defining a basic loop in theano.

        Parameters
        ----------
        inputs : sequence of theano expressions
            Inputs to the scan operation.
        outputs : sequence of output specifiers
            Specifiers for the outputs of the scan operation. This should be a
            sequence containing:
            - None for values that are output by the scan but not tapped as
              inputs,
            - an integer or theano scalar (``ndim == 0``) indicating the batch
              size for initial zero state,
            - a theano tensor variable (``ndim > 0``) containing initial state
              data, or
            - a dictionary containing a full output specifier. See
              ``outputs_info`` in the Theano documentation for ``scan``.
        name : str, optional
            Name of the scan variable to create. Defaults to ``'scan'``.
        step : callable, optional
            The callable to apply in the loop. Defaults to :func:`self._step`.
        constants : sequence of tensor, optional
            A sequence of parameters, if any, needed by the step function.

        Returns
        -------
        output(s) : theano expression(s)
            Theano expression(s) representing output(s) from the scan.
        updates : sequence of update tuples
            A sequence of updates to apply inside a theano function.
        '''
        init = []
        for i, x in enumerate(outputs):
            ndim = getattr(x, 'ndim', -1)
            if x is None or isinstance(x, dict) or ndim > 0:
                init.append(x)
                continue
            if isinstance(x, int) or ndim == 0:
                init.append(TT.repeat(theano.shared(
                    np.zeros((1, self.size), util.FLOAT),
                    name=self._fmt('init{}'.format(i))), x, axis=0))
                continue
            raise ValueError('cannot handle input {} for scan!'.format(x))
        return theano.scan(
            step or self._step,
            name=self._fmt(name),
            sequences=inputs,
            outputs_info=init,
            non_sequences=constants,
            go_backwards='back' in self.kwargs.get('direction', '').lower(),
            truncate_gradient=self.kwargs.get('bptt_limit', -1),
        )

    def _create_rates(self, eps=1e-4):
        '''Create a rate parameter (usually for a recurrent network layer).

        Parameters
        ----------
        eps : float, optional
            A "buffer" preventing rate values from getting too close to 0 or 1.
            Defaults to 1e-4.

        Returns
        -------
        rates : theano shared or None
            A vector of rate parameters for certain types of recurrent layers.
        '''
        if self.rate == 'uniform':
            z = np.random.uniform(eps, 1 - eps, size=self.size).astype(util.FLOAT)
            return theano.shared(z, name=self._fmt('rate'))
        if self.rate == 'log':
            z = np.random.uniform(-6, -eps, size=self.size).astype(util.FLOAT)
            return theano.shared(np.exp(z), name=self._fmt('rate'))
        return None

    def to_spec(self):
        '''Create a specification dictionary for this layer.

        Returns
        -------
        spec : dict
            A dictionary specifying the configuration of this layer.
        '''
        spec = super(Recurrent, self).to_spec()
        spec.update(h_0=self.h_0)
        return spec


class RNN(Recurrent):
    r'''Standard recurrent network layer.

    Notes
    -----

    There are many different styles of recurrent network layers, but the one
    implemented here is known as an Elman layer or an SRN (Simple Recurrent
    Network) -- the output from the layer at the previous time step is
    incorporated into the input of the layer at the current time step.

    .. math::
       h_t = \sigma(x_t W_{xh} + h_{t-1} W_{hh} + b)

    Here, :math:`\sigma(\cdot)` is the :ref:`activation function <activations>`
    of the layer, and the subscript represents the time step of the data being
    processed. The state of the hidden layer at time :math:`t` depends on the
    input at time :math:`t` and the state of the hidden layer at time
    :math:`t-1`.

    *Parameters*

    - ``b`` --- bias
    - ``xh`` --- matrix connecting inputs to hiddens
    - ``hh`` --- matrix connecting hiddens to hiddens

    *Outputs*

    - ``out`` --- the post-activation state of the layer
    - ``pre`` --- the pre-activation state of the layer
    '''

    def setup(self):
        '''Set up the parameters and initial values for this layer.'''
        self.add_weights('xh', self.input_size, self.size)
        self.add_weights('hh', self.size, self.size)
        self.add_bias('b', self.size)

    def transform(self, inputs):
        '''Transform the inputs for this layer into an output for the layer.'''
        # input is:   (batch, time, input)
        # scan wants: (time, batch, input)
        i = self._only_input(inputs).dimshuffle(1, 0, 2)
        x = TT.dot(i, self.find('xh')) + self.find('b')

        # output is:  (time, batch, output)
        # we want:    (batch, time, output)
        (p, o), updates = self._scan(
            [x], [None, inputs.get(self.h_0, x.shape[1])])
        pre = p.dimshuffle(1, 0, 2)
        out = o.dimshuffle(1, 0, 2)

        return dict(pre=pre, out=out), updates

    def _step(self, x_t, h_tm1):
        pre = x_t + TT.dot(h_tm1, self.find('hh'))
        return [pre, self.activate(pre)]


class RRNN(Recurrent):
    r'''An RNN with an update rate for each unit.

    Parameters
    ----------
    rate : str, optional
        This parameter controls how rates are represented in the layer. If this
        is ``None``, the default, then rates are computed as a function of the
        input at each time step. If this parameter is ``'vector'``, then rates
        are represented as a single vector of learnable rates. If this parameter
        is ``'uniform'`` then rates are chosen randomly at uniform from the open
        interval (0, 1). If this parameter is ``'log'`` then rates are chosen
        randomly from a log-uniform distribution such that few rates are near 0
        and many rates are near 1.

    Notes
    -----

    In a normal RNN, a hidden unit is updated completely at each time step,
    :math:`h_t = f(x_t, h_{t-1})`. With an explicit update rate, the state of a
    hidden unit is computed as a mixture of the new and old values,

    .. math::
       h_t = (1 - z_t) \odot h_{t-1} + z_t \odot f(x_t, h_{t-1})

    where :math:`\odot` indicates elementwise multiplication.

    Rates might be defined in a number of ways, spanning a continuum between
    vanilla RNNs (i.e., all rate parameters are effectively fixed at 1), fixed
    but non-uniform rates for each hidden unit [Ben12]_, parametric rates that
    are dependent only on the input, all the way to parametric rates that are
    computed as a function of the inputs and the hidden state at each time step
    (i.e., something more like the :class:`gated recurrent unit <GRU>`).

    This class represents rates in different ways depending on the value of the
    ``rate`` parameter at inititialization.

    *Parameters*

    - ``b`` --- vector of bias values for each hidden unit
    - ``xh`` --- matrix connecting inputs to hidden units
    - ``hh`` --- matrix connecting hiddens to hiddens

    If ``rate`` is initialized to the string ``'vector'``, we define:

    - ``r`` --- vector of rates for each hidden unit

    If ``rate`` is initialized to ``None``, we define:

    - ``r`` --- vector of rate bias values for each hidden unit
    - ``xr`` --- matrix connecting inputs to rate values for each hidden unit

    *Outputs*

    - ``out`` --- the post-activation state of the layer
    - ``pre`` --- the pre-activation state of the layer
    - ``hid`` --- the pre-rate-mixing hidden state
    - ``rate`` --- the rate values

    References
    ----------

    .. [Ben12] Y. Bengio, N. Boulanger-Lewandowski, & R. Pascanu. (2012)
       "Advances in Optimizing Recurrent Networks."
       http://arxiv.org/abs/1212.0901

    .. [Jag07] H. Jaeger, M. Lukoševičius, D. Popovici, & U. Siewert. (2007)
       "Optimization and applications of echo state networks with
       leaky-integrator neurons." Neural Networks, 20(3):335–352.
    '''

    def __init__(self, rate='matrix', **kwargs):
        self.rate = rate.lower().strip()
        super(RRNN, self).__init__(**kwargs)
        self._rates = self._create_rates()

    def setup(self):
        '''Set up the parameters and initial values for this layer.'''
        self.add_weights('xh', self.input_size, self.size)
        self.add_weights('hh', self.size, self.size)
        self.add_bias('b', self.size)

        if self.rate == 'vector' or self.rate == 'matrix':
            self.add_bias('r', self.size)
            if self.rate == 'matrix':
                self.add_weights('xr', self.input_size, self.size)

    def transform(self, inputs):
        '''Transform the inputs for this layer into an output for the layer.'''
        # input is:   (batch, time, input)
        # scan wants: (time, batch, input)
        x = self._only_input(inputs).dimshuffle(1, 0, 2)

        step = self._step_static
        arrays = [TT.dot(x, self.find('xh')) + self.find('b')]
        const = []
        if self.rate == 'matrix':
            step = self._step_dynamic
            r = TT.nnet.sigmoid(TT.dot(x, self.find('xr')) + self.find('r'))
            arrays.append(r)
        elif self.rate == 'vector':
            r = TT.nnet.sigmoid(self.find('r'))
            const.append(r)
        else:
            r = self._rates
            const.append(r)

        # output is:  (time, batch, output)
        # we want:    (batch, time, output)
        (p, h, o), updates = self._scan(
            arrays, [None, None, inputs.get(self.h_0, x.shape[1])],
            constants=const, step=step)
        pre = p.dimshuffle(1, 0, 2)
        hid = h.dimshuffle(1, 0, 2)
        out = o.dimshuffle(1, 0, 2)

        return dict(pre=pre, hid=hid, rate=r, out=out), updates

    def _step_dynamic(self, x_t, r_t, h_tm1):
        pre = x_t + TT.dot(h_tm1, self.find('hh'))
        h_t = self.activate(pre)
        return [pre, h_t, (1 - r_t) * h_tm1 + r_t * h_t]

    def _step_static(self, x_t, h_tm1, r):
        pre = x_t + TT.dot(h_tm1, self.find('hh'))
        h_t = self.activate(pre)
        return [pre, h_t, (1 - r) * h_tm1 + r * h_t]


class MRNN(Recurrent):
    r'''A recurrent network layer with multiplicative dynamics.

    Notes
    -----

    The formulation of MRNN implemented here uses a factored dynamics matrix. To
    understand the motivation for a factored dynamics, imagine for a moment a
    vanilla recurrent layer with one binary input, whose hidden dynamics depend
    on the input, so that :math:`W_{hh}^0` is used if the input is 0, and
    :math:`W_{hh}^1` is used if the input is 1:

    .. math::
       h_t = \sigma(h_{t-1} W_{hh}^{x_t} + x_t W_{xh} + b)

    This generalizes to the idea that there might be an entire collection of
    :math:`W_{hh}^i` matrices that govern the hidden dynamics of the network,
    one for each :math:`0 \le i < N`. But in the general case, it would be
    prohibitively expensive to store this weight tensor; in addition, there are
    probably many shared hidden dynamics that one might want to learn across all
    of these runtime "modes."

    The MRNN solves this problem by factoring the weight tensor idea into two
    2--dimensional arrays. The hidden state is mapped to and from "factor space"
    by :math:`W_{hf}` and :math:`W_{fh}`, respectively, and the latent factors
    are modulated by the input using :math:`W_{xf}`.

    The overall hidden activation for the MRNN model, then, looks like:

    .. math::
       h_t = \sigma((x_t W_{xf} \odot h_{t-1} W_{hf}) W_{fh} + x_t W_{xh} + b)

    where :math:`odot` represents the elementwise product of two vectors.

    *Parameters*

    - ``b`` --- vector of bias values for each hidden unit
    - ``xf`` --- matrix connecting inputs to factors
    - ``xh`` --- matrix connecting inputs to hiddens
    - ``hf`` --- matrix connecting hiddens to factors
    - ``fh`` --- matrix connecting factors to hiddens

    *Outputs*

    - ``out`` --- the post-activation state of the layer
    - ``pre`` --- the pre-activation state of the layer
    - ``factors`` --- the activations of the latent factors

    References
    ----------

    .. [Sut11] I. Sutskever, J. Martens, & G. E. Hinton. (ICML 2011) "Generating
       text with recurrent neural networks."
       http://www.icml-2011.org/papers/524_icmlpaper.pdf
    '''

    def __init__(self, factors=None, **kwargs):
        self.factors = factors or int(np.ceil(np.sqrt(kwargs['size'])))
        super(MRNN, self).__init__(**kwargs)

    def setup(self):
        '''Set up the parameters and initial values for this layer.'''
        self.add_weights('xh', self.input_size, self.size)
        self.add_weights('xf', self.input_size, self.factors)
        self.add_weights('hf', self.size, self.factors)
        self.add_weights('fh', self.factors, self.size)
        self.add_bias('b', self.size)

    def transform(self, inputs):
        '''Transform the inputs for this layer into an output for the layer.'''
        # input is:   (batch, time, input)
        # scan wants: (time, batch, input)
        x = self._only_input(inputs).dimshuffle(1, 0, 2)
        h = TT.dot(x, self.find('xh')) + self.find('b')
        f = TT.dot(x, self.find('xf'))

        # output is:  (time, batch, output)
        # we want:    (batch, time, output)
        (p, o), updates = self._scan(
            [h, f], [None, inputs.get(self.h_0, x.shape[1])])
        pre = p.dimshuffle(1, 0, 2)
        out = o.dimshuffle(1, 0, 2)

        return dict(pre=pre, factors=f, out=out), updates

    def _step(self, x_t, f_t, h_tm1):
        pre = x_t + TT.dot(f_t * TT.dot(h_tm1, self.find('hf')), self.find('fh'))
        return [pre, self.activate(pre)]

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
    r'''Long Short-Term Memory (LSTM) layer.

    An LSTM layer is composed of a number of "cells" that are explicitly
    designed to store information for a certain period of time. Each cell's
    stored value is "guarded" by three gates that permit or deny modification of
    the cell's value:

    - The "input" gate turns on when the input to the LSTM layer should
      influence the cell's value.
    - The "output" gate turns on when the cell's stored value should propagate
      to the next layer.
    - The "forget" gate turns on when the cell's stored value should be reset.

    Notes
    -----

    The output :math:`h_t` of the LSTM layer at time :math:`t` is given as a
    function of the input :math:`x_t` and the previous states of the layer
    :math:`h_{t-1}` and the internal cell :math:`c_{t-1}` by:

    .. math::
       \begin{eqnarray}
       i_t &=& \sigma(x_t W_{xi} + h_{t-1} W_{hi} + c_{t-1} W_{ci} + b_i) \\
       f_t &=& \sigma(x_t W_{xf} + h_{t-1} W_{hf} + c_{t-1} W_{cf} + b_f) \\
       c_t &=& f_t c_{t-1} + i_t \tanh(x_t W_{xc} + h_{t-1} W_{hc} + b_c) \\
       o_t &=& \sigma(x_t W_{xo} + h_{t-1} W_{ho} + c_t W_{co} + b_o) \\
       h_t &=& o_t \tanh(c_t)
       \end{eqnarray}

    where the :math:`W_{ab}` are weight matrix parameters and the :math:`b_x`
    are bias vectors. Equations (1), (2), and (4) give the activations for the
    three gates in the LSTM unit; these gates are activated using the logistic
    sigmoid so that their activities are confined to the open interval (0, 1).
    The value of the cell is updated by equation (3) and is just the weighted
    sum of the previous cell value and the new cell value, where the weights are
    given by the forget and input gate activations, respectively. The output of
    the unit is the cell value weighted by the activation of the output gate.

    The LSTM cell has become quite popular in recurrent neural network models.
    It works amazingly well across a wide variety of tasks and is relatively
    stable during training. The cost of this performance comes in the form of
    large numbers of trainable parameters: Each gate as well as the cell
    receives input from the current input, the previous state of all cells in
    the LSTM layer, and the previous output of the LSTM layer.

    The implementation details for this layer come from the specification given
    on page 5 of [Gra13a]_.

    *Parameters*

    - ``b`` --- vector of bias values for each hidden unit
    - ``ci`` --- vector of peephole input weights
    - ``cf`` --- vector of peephole forget weights
    - ``co`` --- vector of peephole output weights
    - ``xh`` --- matrix connecting inputs to four gates
    - ``hh`` --- matrix connecting hiddens to four gates

    *Outputs*

    - ``out`` --- the post-activation state of the layer
    - ``cell`` --- the state of the hidden "cell"

    Examples
    --------

    LSTM layers can be incorporated into classification models:

    >>> cls = theanets.recurrent.Classifier((28, (100, 'lstm'), 10))

    or regression models:

    >>> reg = theanets.recurrent.Regressor((28, dict(size=100, form='lstm'), 10))

    This layer's parameters can be retrieved using :func:`find
    <theanets.layers.base.Layer.find>`:

    >>> bias = net.find('hid1', 'b')
    >>> ci = net.find('hid1', 'ci')

    References
    ----------

    .. [Hoc97] S. Hochreiter & J. Schmidhuber. (1997) "Long short-term memory."
       Neural computation, 9(8), 1735-1780.

    .. [Gra13a] A. Graves. (2013) "Generating Sequences with Recurrent Neural
       Networks." http://arxiv.org/pdf/1308.0850v5.pdf
    '''

    def __init__(self, c_0=None, **kwargs):
        super(LSTM, self).__init__(**kwargs)
        self.c_0 = c_0

    def resolve(self, layers):
        '''Resolve the names of inputs for this layer.'''
        super(LSTM, self).resolve(layers)
        self.c_0 = self._resolve(self.c_0, layers)

    def setup(self):
        '''Set up the parameters and initial values for this layer.'''
        self.add_weights('xh', self.input_size, 4 * self.size)
        self.add_weights('hh', self.size, 4 * self.size)
        self.add_bias('b', 4 * self.size, mean=2)
        # the three "peephole" weight matrices are always diagonal.
        self.add_bias('ci', self.size)
        self.add_bias('cf', self.size)
        self.add_bias('co', self.size)

    def transform(self, inputs):
        '''Transform the inputs for this layer into an output for the layer.'''
        # input is:   (batch, time, input)
        # scan wants: (time, batch, input)
        x = self._only_input(inputs).dimshuffle(1, 0, 2)

        (o, c), updates = self._scan(
            [TT.dot(x, self.find('xh')) + self.find('b')],
            [inputs.get(self.h_0, x.shape[1]), inputs.get(self.c_0, x.shape[1])])

        # output is:  (time, batch, output)
        # we want:    (batch, time, output)
        out = o.dimshuffle(1, 0, 2)
        cell = c.dimshuffle(1, 0, 2)

        return dict(out=out, cell=cell), updates

    def _step(self, x_t, h_tm1, c_tm1):
        def split(z):
            n = self.size
            return z[:, 0*n:1*n], z[:, 1*n:2*n], z[:, 2*n:3*n], z[:, 3*n:4*n]
        xi, xf, xc, xo = split(x_t + TT.dot(h_tm1, self.find('hh')))
        i_t = TT.nnet.sigmoid(xi + c_tm1 * self.find('ci'))
        f_t = TT.nnet.sigmoid(xf + c_tm1 * self.find('cf'))
        c_t = f_t * c_tm1 + i_t * TT.tanh(xc)
        o_t = TT.nnet.sigmoid(xo + c_t * self.find('co'))
        h_t = o_t * TT.tanh(c_t)
        return [h_t, c_t]

    def to_spec(self):
        '''Create a specification dictionary for this layer.

        Returns
        -------
        spec : dict
            A dictionary specifying the configuration of this layer.
        '''
        spec = super(LSTM, self).to_spec()
        spec.update(c_0=self.c_0)
        return spec


class GRU(Recurrent):
    r'''Gated Recurrent Unit layer.

    Notes
    -----

    The Gated Recurrent Unit lies somewhere between the :class:`LSTM` and the
    :class:`RRNN` in complexity. Like the :class:`RRNN`, its hidden state is
    updated at each time step to be a linear interpolation between the previous
    hidden state, :math:`h_{t-1}`, and the "target" hidden state, :math:`h_t`.
    The interpolation is modulated by an "update gate" that serves the same
    purpose as the rate gates in the :class:`RRNN`. Like the :class:`LSTM`, the
    target hidden state can also be reset using a dedicated gate. All gates in
    this layer are activated based on the current input as well as the previous
    hidden state.

    The update equations in this layer are largely those given by [Chu14]_, page
    4, except for the addition of a hidden bias term. They are:

    .. math::
       \begin{eqnarray}
       r_t &=& \sigma(x_t W_{xr} + h_{t-1} W_{hr} + b_r) \\
       z_t &=& \sigma(x_t W_{xz} + h_{t-1} W_{hz} + b_z) \\
       \hat{h}_t &=& g\left(x_t W_{xh} + (r_t \odot h_{t-1}) W_{hh} + b_h\right) \\
       h_t &=& (1 - z_t) \odot h_{t-1} + z_t \odot \hat{h}_t.
       \end{eqnarray}

    Here, :math:`g(\cdot)` is the activation function for the layer, and
    :math:`\sigma(\cdot)` is the logistic sigmoid, which ensures that the two
    gates in the layer are limited to the open interval (0, 1). The symbol
    :math:`\odot` indicates elementwise multiplication.

    *Parameters*

    - ``bh`` --- vector of bias values for each hidden unit
    - ``br`` --- vector of reset biases
    - ``bz`` --- vector of rate biases
    - ``xh`` --- matrix connecting inputs to hidden units
    - ``xr`` --- matrix connecting inputs to reset gates
    - ``xz`` --- matrix connecting inputs to rate gates
    - ``hh`` --- matrix connecting hiddens to hiddens
    - ``hr`` --- matrix connecting hiddens to reset gates
    - ``hz`` --- matrix connecting hiddens to rate gates

    *Outputs*

    - ``out`` --- the post-activation state of the layer
    - ``pre`` --- the pre-activation state of the layer
    - ``hid`` --- the pre-rate-mixing hidden state
    - ``rate`` --- the rate values

    References
    ----------

    .. [Chu14] J. Chung, C. Gulcehre, K. H. Cho, & Y. Bengio (2014), "Empirical
       Evaluation of Gated Recurrent Neural Networks on Sequence Modeling"
       http://arxiv.org/abs/1412.3555v1
    '''

    def setup(self):
        '''Set up the parameters and initial values for this layer.'''
        self.add_weights('xh', self.input_size, self.size)
        self.add_weights('xr', self.input_size, self.size)
        self.add_weights('xz', self.input_size, self.size)
        self.add_weights('hh', self.size, self.size)
        self.add_weights('hr', self.size, self.size)
        self.add_weights('hz', self.size, self.size)
        self.add_bias('bh', self.size)
        self.add_bias('br', self.size)
        self.add_bias('bz', self.size)

    def transform(self, inputs):
        '''Transform the inputs for this layer into an output for the layer.'''
        # input is:   (batch, time, input)
        # scan wants: (time, batch, input)
        x = self._only_input(inputs).dimshuffle(1, 0, 2)

        (p, h, r, o), updates = self._scan(
            [TT.dot(x, self.find('xh')) + self.find('bh'),
             TT.dot(x, self.find('xr')) + self.find('br'),
             TT.dot(x, self.find('xz')) + self.find('bz')],
            [None, None, None, inputs.get(self.h_0, x.shape[1])])

        # output is:  (time, batch, output)
        # we want:    (batch, time, output)
        pre = p.dimshuffle(1, 0, 2)
        hid = h.dimshuffle(1, 0, 2)
        rate = r.dimshuffle(1, 0, 2)
        out = o.dimshuffle(1, 0, 2)

        return dict(pre=pre, hid=hid, rate=rate, out=out), updates

    def _step(self, x_t, r_t, z_t, h_tm1):
        r = TT.nnet.sigmoid(r_t + TT.dot(h_tm1, self.find('hr')))
        z = TT.nnet.sigmoid(z_t + TT.dot(h_tm1, self.find('hz')))
        pre = x_t + TT.dot(r * h_tm1, self.find('hh'))
        h_t = self.activate(pre)
        return [pre, h_t, z, (1 - z) * h_tm1 + z * h_t]


class Clockwork(Recurrent):
    r'''A Clockwork RNN layer updates "modules" of neurons at specific rates.

    Notes
    -----

    In a vanilla :class:`RNN` layer, all neurons in the hidden pool are updated
    at every time step by mixing an affine transformation of the input with an
    affine transformation of the state of the hidden pool neurons at the
    previous time step:

    .. math::
       h_t = g(x_tW_{xh} + h_{t-1}W_{hh} + b_h)

    In a Clockwork RNN layer, neurons in the hidden pool are split into
    :math:`M` "modules" of equal size (:math:`h^i` for :math:`i = 1, \dots, M`),
    each of which has an associated clock period (a positive integer :math:`T_i`
    for :math:`i = 1, \dots, M`). The neurons in module :math:`i` are updated
    only when the time index :math:`t` of the input :math:`x_t` is an even
    multiple of :math:`T_i`. Thus some of modules (those with large :math:`T`)
    only respond to "slow" features in the input, and others (those with small
    :math:`T`) respond to "fast" features.

    Furthermore, "fast" modules with small periods receive inputs from "slow"
    modules with large periods, but not vice-versa: this allows the "slow"
    features to influence the "fast" features, but not the other way around.

    The state :math:`h_t^i` of module :math:`i` at time step :math:`t` is thus
    governed by the following mathematical relation:

    .. math::
       h_t^i = \left\{ \begin{align*}
          &g\left( x_tW_{xh}^i + b_h^i +
             \sum_{j=i}^M h_{t-1}^jW_{hh}^j\right)
             \mbox{ if } t \mod T_i = 0 \\
          &h_{t-1}^i \mbox{ otherwise.} \end{align*} \right.

    Here, the :math:`M` modules have been ordered such that :math:`T_i < T_j`
    for :math:`i < j` -- that is, the modules are ordered from "fastest" to
    "slowest."

    Note that, unlike in the original paper, the hidden-hidden weight matrix is
    stored in full (i.e., it is ``size`` x ``size``); the module separation is
    enforced by masking this weight matrix with zeros in the appropriate places.
    This implementation runs *much* faster on a GPU than an approach that uses
    dedicated module parameters.

    *Parameters*

    - ``b`` --- vector of bias values for each hidden unit
    - ``xh`` --- matrix connecting inputs to hidden units
    - ``hh`` --- matrix connecting hiddens to hiddens

    *Outputs*

    - ``out`` --- the post-activation state of the layer
    - ``pre`` --- the pre-activation state of the layer

    Parameters
    ----------
    periods : sequence of int
        The periods for the modules in this clockwork layer. The number of
        values in this sequence specifies the number of modules in the layer.
        The layer size must be an integer multiple of the number of modules
        given in this sequence.

    References
    ----------

    .. [Kou14] J. Koutník, K. Greff, F. Gomez, & J. Schmidhuber. (2014) "A
       Clockwork RNN." http://arxiv.org/abs/1402.3511
    '''

    def __init__(self, periods, **kwargs):
        assert kwargs['size'] % len(periods) == 0
        self.periods = np.asarray(sorted(periods))
        super(Clockwork, self).__init__(**kwargs)

    def setup(self):
        '''Set up the parameters and initial values for this layer.'''
        n = self.size // len(self.periods)
        mask = np.zeros((self.size, self.size), util.FLOAT)
        period = np.zeros((self.size, ), 'i')
        for i, T in enumerate(self.periods):
            # see https://github.com/lmjohns3/theanets/issues/125
            mask[i*n:, i*n:(i+1)*n] = 1
            period[i*n:(i+1)*n] = T
        self._mask = theano.shared(mask, name='mask')
        self._period = theano.shared(period, name='period')
        self.add_weights('hh', self.size, self.size)
        self.add_weights('xh', self.input_size, self.size)
        self.add_bias('b', self.size)

    def log(self):
        '''Log some information about this layer.'''
        inputs = ', '.join('({0}){1.size}'.format(n, l)
                           for n, l in self._resolved_inputs.items())
        logging.info('layer %s "%s": %s -> %s, [%s] %s, %d parameters',
                     self.__class__.__name__,
                     self.name,
                     inputs,
                     self.size,
                     ' '.join(str(T) for T in self.periods),
                     getattr(self.activate, 'name', self.activate),
                     sum(np.prod(p.get_value().shape) for p in self.params))

    def transform(self, inputs):
        '''Transform the inputs for this layer into an output for the layer.'''
        # input is:   (batch, time, input)
        # scan wants: (time, batch, input)
        i = self._only_input(inputs).dimshuffle(1, 0, 2)
        x = TT.dot(i, self.find('xh')) + self.find('b')

        init = inputs.get(self.h_0, x.shape[1])

        # output is:  (time, batch, output)
        # we want:    (batch, time, output)
        (p, o), updates = self._scan([TT.arange(x.shape[0]), x], [init, init])
        pre = p.dimshuffle(1, 0, 2)
        out = o.dimshuffle(1, 0, 2)

        return dict(pre=pre, out=out), updates

    def _step(self, t, x_t, pre_tm1, h_tm1):
        pre = x_t + TT.dot(h_tm1, self.find('hh') * self._mask)
        pre_t = TT.switch(TT.eq(t % self._period, 0), pre, pre_tm1)
        return [pre_t, self.activate(pre_t)]

    def to_spec(self):
        '''Create a specification dictionary for this layer.

        Returns
        -------
        spec : dict
            A dictionary specifying the configuration of this layer.
        '''
        spec = super(Clockwork, self).to_spec()
        spec['periods'] = tuple(self.periods)
        return spec


class MUT1(Recurrent):
    r'''"MUT1" evolved recurrent layer.

    Notes
    -----

    This layer is a close cousin of the :class:`GRU`, which updates the state of
    the hidden units by linearly interpolating the state from the previous time
    step with a "target" state. Unlike the GRU, however, this layer omits a
    dependency on the hidden state for the "rate gate", and the current input is
    piped through the tanh function before influencing the target hidden state.

    The update equations in this layer are mostly those given by [Joz15]_, page
    7:

    .. math::
       \begin{eqnarray}
       r_t &=& \sigma(x_t W_{xr} + h_{t-1} W_{hr} + b_r) \\
       z_t &=& \sigma(x_t W_{xz} + b_z) \\
       \hat{h}_t &=& \tanh\left(\tanh(x_t W_{xh}) +
          (r_t \odot h_{t-1}) W_{hh} + b_h\right) \\
       h_t &=& (1 - z_t) \odot h_{t-1} + z_t \odot \hat{h}_t.
       \end{eqnarray}

    Here, the layer activation is always set to :math:`\tanh`, and
    :math:`\sigma(\cdot)` is the logistic sigmoid, which ensures that the two
    gates in the layer are limited to the open interval (0, 1). The symbol
    :math:`\odot` indicates elementwise multiplication.

    *Parameters*

    - ``bh`` --- vector of bias values for each hidden unit
    - ``br`` --- vector of reset biases
    - ``bz`` --- vector of rate biases
    - ``xh`` --- matrix connecting inputs to hidden units
    - ``xr`` --- matrix connecting inputs to reset gates
    - ``xz`` --- matrix connecting inputs to rate gates
    - ``hh`` --- matrix connecting hiddens to hiddens
    - ``hr`` --- matrix connecting hiddens to reset gates

    *Outputs*

    - ``out`` --- the post-activation state of the layer
    - ``pre`` --- the pre-activation state of the layer
    - ``hid`` --- the pre-rate-mixing hidden state
    - ``rate`` --- the rate values

    References
    ----------

    .. [Joz15] R. Jozefowicz, W. Zaremba, & I. Sutskever (2015) "An Empirical
       Exploration of Recurrent Network Architectures."
       http://jmlr.org/proceedings/papers/v37/jozefowicz15.pdf
    '''

    def setup(self):
        '''Set up the parameters and initial values for this layer.'''
        self.add_weights('xh', self.input_size, self.size)
        self.add_weights('xr', self.input_size, self.size)
        self.add_weights('xz', self.input_size, self.size)
        self.add_weights('hh', self.size, self.size)
        self.add_weights('hr', self.size, self.size)
        self.add_bias('bh', self.size)
        self.add_bias('br', self.size)
        self.add_bias('bz', self.size)

    def transform(self, inputs):
        '''Transform the inputs for this layer into an output for the layer.'''
        # input is:   (batch, time, input)
        # scan wants: (time, batch, input)
        x = self._only_input(inputs).dimshuffle(1, 0, 2)
        z = TT.nnet.sigmoid(TT.dot(x, self.find('xz')) + self.find('bz'))

        (p, h, o), updates = self._scan(
            [TT.tanh(TT.dot(x, self.find('xh')) + self.find('bh')),
             TT.dot(x, self.find('xr')) + self.find('br'), z],
            [None, None, inputs.get(self.h_0, x.shape[1])])

        # output is:  (time, batch, output)
        # we want:    (batch, time, output)
        pre = p.dimshuffle(1, 0, 2)
        hid = h.dimshuffle(1, 0, 2)
        rate = z.dimshuffle(1, 0, 2)
        out = o.dimshuffle(1, 0, 2)

        return dict(pre=pre, hid=hid, rate=rate, out=out), updates

    def _step(self, x_t, r_t, z_t, h_tm1):
        r = TT.nnet.sigmoid(r_t + TT.dot(h_tm1, self.find('hr')))
        pre = x_t + TT.dot(r * h_tm1, self.find('hh'))
        h_t = TT.tanh(pre)
        return [pre, h_t, (1 - z_t) * h_tm1 + z_t * h_t]


class SCRN(Recurrent):
    r'''Simple Contextual Recurrent Network layer.

    Notes
    -----

    A Simple Contextual Recurrent Network incorporates an explicitly slow-moving
    hidden context layer with a simple recurrent network.

    The update equations in this layer are largely those given by [Mik15]_,
    pages 4 and 5, but this implementation adds a bias term for the output of
    the layer. The update equations are thus:

    .. math::
       \begin{eqnarray}
       s_t &=& r \odot x_t W_{xs} + (1 - r) \odot s_{t-1} \\
       h_t &=& \sigma(x_t W_{xh} + h_{t-1} W_{hh} + s_t W_{sh}) \\
       o_t &=& g\left(h_t W_{ho} + s_t W_{so} + b\right). \\
       \end{eqnarray}

    Here, :math:`g(\cdot)` is the activation function for the layer and
    :math:`\odot` is elementwise multiplication. The rate values :math:`r` are
    computed using :math:`r = \sigma(\hat{r})` so that the rate values are
    limited to the open interval (0, 1). :math:`\sigma(\cdot)` is the logistic
    sigmoid.

    *Parameters*

    - ``xs`` --- matrix connecting inputs to state units (called B in the paper)
    - ``xh`` --- matrix connecting inputs to hidden units (A)
    - ``sh`` --- matrix connecting state to hiddens (P)
    - ``hh`` --- matrix connecting hiddens to hiddens (R)
    - ``ho`` --- matrix connecting hiddens to output (U)
    - ``so`` --- matrix connecting state to output (V)
    - ``b`` --- vector of output bias values (not in original paper)

    Additionally, if ``rate`` is specified as ``'vector'`` (the default), then
    we also have:

    - ``r`` --- vector of learned rate values for the state units

    *Outputs*

    - ``out`` --- the post-activation state of the layer
    - ``pre`` --- the pre-activation state of the layer
    - ``hid`` --- the state of the layer's hidden units
    - ``state`` --- the state of the layer's state units
    - ``rate`` --- the rate values of the state units

    References
    ----------

    .. [Mik15] T. Mikolov, A. Joulin, S. Chopra, M. Mathieu, & M. Ranzato (ICLR
       2015) "Learning Longer Memory in Recurrent Neural Networks."
       http://arxiv.org/abs/1412.7753
    '''

    def __init__(self, rate='vector', s_0=None, **kwargs):
        self.rate = rate.lower().strip()
        super(SCRN, self).__init__(**kwargs)
        self._rates = self._create_rates()
        self.s_0 = s_0

    def resolve(self, layers):
        '''Resolve the names of inputs for this layer.'''
        super(SCRN, self).resolve(layers)
        self.s_0 = self._resolve(self.s_0, layers)

    def setup(self):
        '''Set up the parameters and initial values for this layer.'''
        self.add_weights('xs', self.input_size, self.size)
        self.add_weights('xh', self.input_size, self.size)
        self.add_weights('sh', self.size, self.size)
        self.add_weights('hh', self.size, self.size)
        self.add_weights('ho', self.size, self.size)
        self.add_weights('so', self.size, self.size)

        self.add_bias('b', self.size)

        if self.rate == 'vector':
            self.add_bias('r', self.size)

    def transform(self, inputs):
        '''Transform the inputs for this layer into an output for the layer.'''
        # input is:   (batch, time, input)
        # scan wants: (time, batch, input)
        x = self._only_input(inputs).dimshuffle(1, 0, 2)

        r = self._rates
        if self.rate == 'vector':
            r = TT.nnet.sigmoid(self.find('r'))

        (p, _, s), updates = self._scan(
            [TT.dot(x, self.find('xh')), TT.dot(x, self.find('xs'))],
            [None,
             inputs.get(self.h_0, x.shape[1]),
             inputs.get(self.s_0, x.shape[1])],
            constants=[r])

        # output is:  (time, batch, output)
        # we want:    (batch, time, output)
        hid = TT.nnet.sigmoid(p.dimshuffle(1, 0, 2))
        state = s.dimshuffle(1, 0, 2)

        pre = (TT.dot(hid, self.find('ho')) +
               TT.dot(state, self.find('so')) +
               self.find('b'))

        return dict(
            rate=r, state=state, hid=hid, pre=pre, out=self.activate(pre),
        ), updates

    def _step(self, xh_t, xs_t, h_tm1, s_tm1, r):
        s = (1 - r) * s_tm1 + r * xs_t
        p = xh_t + TT.dot(h_tm1, self.find('hh')) + TT.dot(s, self.find('sh'))
        return [p, TT.nnet.sigmoid(p), s]

    def to_spec(self):
        '''Create a specification dictionary for this layer.

        Returns
        -------
        spec : dict
            A dictionary specifying the configuration of this layer.
        '''
        spec = super(SCRN, self).to_spec()
        spec.update(s_0=self.s_0)
        return spec


class Bidirectional(base.Layer):
    r'''A bidirectional recurrent layer runs worker models forward and backward.

    Notes
    -----

    The size of this layer is split in half, with each half allocated to a
    "worker" layer that processes data in one direction in time. The outputs of
    the forward and backward passes are concatenated into the overall output for
    the layer.

    For an example specification of a bidirectional recurrent network, see
    [Gra13b]_.

    Parameters
    ----------
    worker : str, optional
        This string specifies the type of worker layer to use for the forward
        and backward processing. This parameter defaults to 'rnn' (i.e., vanilla
        recurrent network layer), but can be given as any string that specifies
        a recurrent layer type.

    Attributes
    ----------
    worker : str
        The form of the underlying worker networks.
    forward : :class:`theanets.layers.base.Layer`
        The layer that processes input data forwards in time.
    backward : :class:`theanets.layers.base.Layer`
        The layer that processes input data backwards in time.

    References
    ----------

    .. [Gra13b] A. Graves, N. Jaitly, & A. Mohamed. (2013) "Hybrid Speech
       Recognition with Deep Bidirectional LSTM."
       http://www.cs.toronto.edu/~graves/asru_2013.pdf
    '''

    def __init__(self, worker='rnn', **kwargs):
        size = kwargs.pop('size')
        name = kwargs.pop('name', 'layer{}'.format(base.Layer._count))
        if 'direction' in kwargs:
            kwargs.pop('direction')

        def make(suffix, direction):
            return base.Layer.build(
                worker,
                direction=direction,
                size=size // 2,
                name='{}_{}'.format(name, suffix),
                **kwargs)

        self.worker = worker
        self.forward = make('fw', 'forward')
        self.backward = make('bw', 'backward')
        super(Bidirectional, self).__init__(size=size, name=name, **kwargs)

    @property
    def params(self):
        '''A list of all learnable parameters in this layer.'''
        return self.forward.params + self.backward.params

    def bind(self, *args, **kwargs):
        super(Bidirectional, self).bind(*args, **kwargs)
        self.forward.bind(*args, **kwargs)
        self.backward.bind(*args, **kwargs)

    def transform(self, inputs):
        '''Transform the inputs for this layer into an output for the layer.'''
        fout, fupd = self.forward.transform(inputs)
        bout, bupd = self.backward.transform(inputs)
        outputs = dict(out=TT.concatenate([fout['out'], bout['out']], axis=2))
        if 'pre' in fout:
            outputs['pre'] = TT.concatenate([fout['pre'], bout['pre']], axis=2)
        if 'cell' in fout:
            outputs['cell'] = TT.concatenate([fout['cell'], bout['cell']], axis=2)
        for k, v in fout.items():
            outputs['fw_{}'.format(k)] = v
        for k, v in bout.items():
            outputs['bw_{}'.format(k)] = v
        return outputs, fupd + bupd

    def to_spec(self):
        '''Create a specification dictionary for this layer.

        Returns
        -------
        spec : dict
            A dictionary specifying the configuration of this layer.
        '''
        spec = super(Bidirectional, self).to_spec()
        spec['worker'] = self.worker
        return spec

# -*- coding: utf-8 -*-

r'''Recurrent layers for neural network computation graphs.

Recurrent layers are basically defined by the presence of explicitly modeled
time dependencies in the computation graph.
'''

from __future__ import division

import climate
import numpy as np
import theano
import theano.tensor as TT

from . import base
from .. import util

logging = climate.get_logger(__name__)

FLOAT = theano.config.floatX

__all__ = [
    'RNN',
    'ARRNN',
    'LRRNN',
    'GRU',
    'LSTM',
    'Clockwork',
    'MRNN',
    'Bidirectional',
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
                                   diagonal=d, rng=self.nrng)
                for _ in range(nout // nin)], axis=1)
        else:
            arr = util.random_matrix(nin, nout, mean, std, sparsity=s, rng=self.nrng)
        self._params.append(theano.shared(arr, name=self._fmt(name)))

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
            truncate_gradient=self.kwargs.get('bptt_limit', -1),
        )


class RNN(Recurrent):
    r'''Standard recurrent network layer.

    There are many different styles of recurrent network layers, but the one
    implemented here is known as an Elman layer or an SRN (Simple Recurrent
    Network) -- the output from the layer at the previous time step is
    incorporated into the input of the layer at the current time step.
    '''

    def setup(self):
        '''Set up the parameters and initial values for this layer.'''
        self.add_weights('xh', self.input_size, self.size)
        self.add_weights('hh', self.size, self.size)
        self.add_bias('b', self.size)

    def transform(self, inputs):
        '''Transform the inputs for this layer into an output for the layer.

        Parameters
        ----------
        inputs : dict of theano expressions
            Symbolic inputs to this layer, given as a dictionary mapping string
            names to Theano expressions. See :func:`base.Layer.connect`.

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
        x = TT.dot(self._only_input(inputs), self.find('xh')) + self.find('b')
        (pre, out), updates = self._scan(fn, [x], [None, x])
        return dict(pre=pre, out=out), updates


class LRRNN(Recurrent):
    r'''A learned-rate RNN defines per-hidden-unit accumulation rates.

    In a normal RNN, a hidden unit is updated completely at each time step,
    :math:`h_t = f(x_t, h_{t-1})`. With an explicit update rate, the state of a
    hidden unit is computed as a mixture of the new and old values, `h_t =
    \alpha_t h_{t-1} + (1 - \alpha_t) f(x_t, h_{t-1})`.

    Rates might be defined in a number of ways, spanning a continuum between
    vanilla RNNs (i.e., all rate parameters are fixed at 1), fixed but
    non-uniform rates for each hidden unit [Ben12]_, parametric rates that are
    dependent only on the input (i.e., the :class:`ARRNN`), all the way to
    parametric rates that are computed as a function of the inputs and the
    hidden state at each time step (i.e., something more like the :class:`gated
    recurrent unit <GRU>`).

    This class represents rates as a single learnable vector of parameters. This
    representation uses the fewest number of parameters for learnable rates, but
    the simplicity of the model comes at the cost of effectively fixing the rate
    for each unit as a constant value across time.

    References
    ----------

    .. [Ben12] Y. Bengio, N. Boulanger-Lewandowski, R. Pascanu. (2012) "Advances
       in Optimizing Recurrent Networks." http://arxiv.org/abs/1212.0901
    '''

    def setup(self):
        '''Set up the parameters and initial values for this layer.'''
        self.add_weights('xh', self.input_size, self.size)
        self.add_weights('hh', self.size, self.size)
        self.add_bias('b', self.size)
        self.add_bias('r', self.size, mean=2, std=1)

    def transform(self, inputs):
        '''Transform the inputs for this layer into an output for the layer.

        Parameters
        ----------
        inputs : dict of theano expressions
            Symbolic inputs to this layer, given as a dictionary mapping string
            names to Theano expressions. See :func:`base.Layer.connect`.

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
        x = self._only_input(inputs)
        h = TT.dot(x, self.find('xh')) + self.find('b')

        def fn(x_t, h_tm1):
            pre = x_t + TT.dot(h_tm1, self.find('hh'))
            h_t = self.activate(pre)
            return [pre, h_t, r * h_tm1 + (1 - r) * h_t]

        (pre, hid, out), updates = self._scan(fn, [h], [None, None, x])
        return dict(pre=pre, hid=hid, rate=r, out=out), updates


class ARRNN(Recurrent):
    r'''An adaptive-rate RNN defines per-hidden-unit accumulation rates.

    In a normal RNN, a hidden unit is updated completely at each time step,
    :math:`h_t = f(x_t, h_{t-1})`. With an explicit update rate, the state of a
    hidden unit is computed as a mixture of the new and old values, `h_t =
    \alpha_t h_{t-1} + (1 - \alpha_t) f(x_t, h_{t-1})`.

    Rates might be defined in a number of ways, spanning a continuum between
    vanilla RNNs (i.e., all rate parameters are fixed at 1) all the way to
    parametric rates that are computed as a function of the inputs and the
    hidden state at each time step (i.e., something more like the :class:`gated
    recurrent unit <GRU>`).

    In the ARRNN model, the rate values are represented as a computed at each
    time step as a logistic sigmoid applied to an affine transform of the input:
    :math:`\alpha_t = 1 / (1 + e^{-x_t W_{xr} - b_r})`. This representation of
    the rates uses more parameters than the :class:`LRRNN` but is able to adapt
    rates to the input at each time step. However, in this model, rates are not
    able to adapt to the state of the hidden units at each time step.
    '''

    def setup(self):
        '''Set up the parameters and initial values for this layer.'''
        self.add_weights('xh', self.input_size, self.size)
        self.add_weights('xr', self.input_size, self.size)
        self.add_weights('hh', self.size, self.size)
        self.add_bias('b', self.size)
        self.add_bias('r', self.size)

    def transform(self, inputs):
        '''Transform the inputs for this layer into an output for the layer.

        Parameters
        ----------
        inputs : dict of theano expressions
            Symbolic inputs to this layer, given as a dictionary mapping string
            names to Theano expressions. See :func:`base.Layer.connect`.

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
        x = self._only_input(inputs)
        r = TT.nnet.sigmoid(TT.dot(x, self.find('xr')) + self.find('r'))
        h = TT.dot(x, self.find('xh')) + self.find('b')

        def fn(x_t, r_t, h_tm1):
            pre = x_t + TT.dot(h_tm1, self.find('hh'))
            h_t = self.activate(pre)
            return [pre, h_t, r_t * h_tm1 + (1 - r_t) * h_t]

        (pre, hid, out), updates = self._scan(fn, [h, r], [None, None, x])
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
        self.add_weights('xh', self.input_size, self.size)
        self.add_weights('xf', self.input_size, self.factors)
        self.add_weights('hf', self.size, self.factors)
        self.add_weights('fh', self.factors, self.size)
        self.add_bias('b', self.size)

    def transform(self, inputs):
        '''Transform the inputs for this layer into an output for the layer.

        Parameters
        ----------
        inputs : dict of theano expressions
            Symbolic inputs to this layer, given as a dictionary mapping string
            names to Theano expressions. See :func:`base.Layer.connect`.

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
        x = self._only_input(inputs)
        h = TT.dot(x, self.find('xh')) + self.find('b')
        f = TT.dot(x, self.find('xf'))

        def fn(x_t, f_t, h_tm1):
            pre = x_t + TT.dot(f_t * TT.dot(h_tm1, self.find('hf')), self.find('fh'))
            return [pre, self.activate(pre)]

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
    on page 5 of [2]_.

    References
    ----------

    .. [1] S. Hochreiter & J. Schmidhuber. (1997) "Long short-term memory."
           Neural computation, 9(8), 1735-1780.

    .. [2] A. Graves. (2013) "Generating Sequences with Recurrent Neural
           Networks." http://arxiv.org/pdf/1308.0850v5.pdf
    '''

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
        '''Transform the inputs for this layer into an output for the layer.

        Parameters
        ----------
        inputs : dict of theano expressions
            Symbolic inputs to this layer, given as a dictionary mapping string
            names to Theano expressions. See :func:`base.Layer.connect`.

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

        x = self._only_input(inputs)
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
        '''Transform inputs to this layer into outputs for the layer.

        Parameters
        ----------
        inputs : dict of theano expressions
            Symbolic inputs to this layer, given as a dictionary mapping string
            names to Theano expressions. See :func:`base.Layer.connect`.

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
        def fn(x_t, r_t, z_t, h_tm1):
            r = TT.nnet.sigmoid(r_t + TT.dot(h_tm1, self.find('hr')))
            z = TT.nnet.sigmoid(z_t + TT.dot(h_tm1, self.find('hz')))
            pre = x_t + TT.dot(r * h_tm1, self.find('hh'))
            h_t = self.activate(pre)
            return [pre, h_t, z, (1 - z) * h_tm1 + z * h_t]

        x = self._only_input(inputs)
        (pre, hid, rate, out), updates = self._scan(
            fn,
            [TT.dot(x, self.find('xh')) + self.find('bh'),
             TT.dot(x, self.find('xr')) + self.find('br'),
             TT.dot(x, self.find('xz')) + self.find('bz')],
            [None, None, None, x])
        return dict(pre=pre, hid=hid, rate=rate, out=out), updates


class Clockwork(Recurrent):
    r'''A Clockwork RNN layer updates "modules" of neurons at specific rates.

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
             \sum_{j=1}^i h_{t-1}^jW_{hh}^j\right)
             \mbox{ if } t \mod T_i = 0 \\
          &h_{t-1}^i \mbox{ otherwise.} \end{align*} \right.

    Here, the modules have been ordered such that :math:`T_j > T_i` for
    :math:`j < i`.

    In ``theanets``, this update relation is implemented using a nested loop.
    The outer loop calls Theano's ``scan()`` operator to iterate over the input
    data at each time step. The inner loop iterates over the modules, updating
    each module if the clock cycle is correct, and copying over the previous
    value of the module if not.

    Parameters
    ----------
    periods : sequence of int
        The periods for the modules in this clockwork layer. The number of
        values in this sequence specifies the number of modules in the layer.
        The layer size must be an integer multiple of the number of modules
        given in this sequence.

    References
    ----------

    .. [1] J. Koutník, K. Greff, F. Gomez, & J. Schmidhuber. (2014) "A Clockwork
           RNN." http://arxiv.org/abs/1402.3511
    '''

    def __init__(self, periods, **kwargs):
        assert kwargs['size'] % len(periods) == 0
        self.periods = np.asarray(sorted(periods, reverse=True))
        super(Clockwork, self).__init__(**kwargs)

    def setup(self):
        n = self.size // len(self.periods)
        mask = np.zeros((self.size, self.size), FLOAT)
        period = np.zeros((self.size, ), 'i')
        for i, T in enumerate(self.periods):
            mask[i*n:(i+1)*n, i*n:] = 1
            period[i*n:(i+1)*n] = T
        self._mask = theano.shared(mask, name='mask')
        self._period = theano.shared(period, name='period')
        self.add_weights('hh', self.size, self.size)
        self.add_weights('xh', self.input_size, self.size)
        self.add_bias('b', self.size)

    def log(self):
        '''Log some information about this layer.'''
        inputs = ', '.join('({}){}'.format(n, s) for n, s in self.inputs.items())
        logging.info('layer %s "%s": %s -> %s, [%s] %s, %d parameters',
                     self.__class__.__name__,
                     self.name,
                     inputs,
                     self.size,
                     ' '.join(str(T) for T in self.periods),
                     getattr(self.activate, 'name', self.activate),
                     self.num_params)

    def transform(self, inputs):
        '''Transform inputs to this layer into outputs for the layer.

        Parameters
        ----------
        inputs : dict of theano expressions
            Symbolic inputs to this layer, given as a dictionary mapping string
            names to Theano expressions. See :func:`base.Layer.connect`.

        Returns
        -------
        outputs : dict of theano expressions
            A map from string output names to Theano expressions for the outputs
            from this layer. This layer type generates a "pre" output that gives
            the unit activity before applying the layer's activation function,
            and a "hid" output that gives the post-activation values.
        updates : sequence of update pairs
            A sequence of updates to apply to this layer's state inside a theano
            function.
        '''
        def fn(t, x_t, p_tm1, h_tm1):
            p = x_t + TT.dot(h_tm1, self.find('hh') * self._mask)
            p_t = TT.switch(TT.eq(t % self._period, 0), p, p_tm1)
            return [p_t, self.activate(p_t)]

        x = TT.dot(self._only_input(inputs), self.find('xh')) + self.find('b')
        (pre, out), updates = self._scan(fn, [TT.arange(x.shape[0]), x], [x, x])
        return dict(pre=pre, out=out), updates

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


class Bidirectional(base.Layer):
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
        name = kwargs.pop('name', 'layer{}'.format(base.Layer._count))
        if 'direction' in kwargs:
            kwargs.pop('direction')

        def make(suffix, direction):
            return base.build(
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

    @property
    def num_params(self):
        '''Total number of learnable parameters in this layer.'''
        return self.forward.num_params + self.backward.num_params

    def transform(self, inputs):
        '''Transform the inputs for this layer into an output for the layer.

        Parameters
        ----------
        inputs : dict of theano expressions
            Symbolic inputs to this layer, given as a dictionary mapping string
            names to Theano expressions. See :func:`base.Layer.connect`.

        Returns
        -------
        outputs : dict of theano expressions
            Theano expressions representing the output from the layer. This
            layer type produces an "out" output that concatenates the outputs
            from its underlying workers. If present, it also concatenates the
            "pre" and "cell" outputs from the underlying workers. Finally, it
            passes along the individual outputs from its workers using "fw" and
            "bw" prefixes for forward and backward directions.
        updates : list of update pairs
            A list of state updates to apply inside a theano function.
        '''
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

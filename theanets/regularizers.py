# -*- coding: utf-8 -*-

r'''This module contains implementations of common regularizers.

In ``theanets`` regularizers are thought of as additional terms that get
combined with the :class:`Loss <theanets.losses.Loss>` for a model at
optimization time. Regularizer terms in the loss are usually used to "encourage"
a model to behave in a particular way---for example, the pattern and arrangement
of learned features can be changed by including a sparsity (L1-norm) regularizer
on the hidden unit activations, or units can randomly be dropped out (set to
zero) while running the model.
'''

import climate
import fnmatch
import theano.tensor as TT

from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

from . import layers
from . import util

logging = climate.get_logger(__name__)


def from_kwargs(graph, **kwargs):
    '''Construct common regularizers from a set of keyword arguments.

    Keyword arguments not listed below will be passed to
    :func:`Regularizer.build` if they specify the name of a registered
    :class:`Regularizer`.

    Parameters
    ----------
    graph : :class:`theanets.graph.Network`
        A network graph to regularize.

    regularizers : dict or tuple/list of :class:`Regularizer`, optional
        If this is a list or a tuple, the contents of the list will be returned
        as the regularizers. This is to permit custom lists of regularizers to
        be passed easily.

        If this is a dict, its contents will be added to the other keyword
        arguments passed in.

    rng : int or theano RandomStreams, optional
        If an integer is provided, it will be used to seed the random number
        generators for the dropout or noise regularizers. If a theano
        RandomStreams object is provided, it will be used directly. Defaults to
        13.

    input_dropout : float, optional
        Apply dropout to input layers in the network graph, with this dropout
        rate. Defaults to 0 (no dropout).

    hidden_dropout : float, optional
        Apply dropout to hidden layers in the network graph, with this dropout
        rate. Defaults to 0 (no dropout).

    output_dropout : float, optional
        Apply dropout to the output layer in the network graph, with this
        dropout rate. Defaults to 0 (no dropout).

    input_noise : float, optional
        Apply noise to input layers in the network graph, with this standard
        deviation. Defaults to 0 (no noise).

    hidden_noise : float, optional
        Apply noise to hidden layers in the network graph, with this standard
        deviation. Defaults to 0 (no noise).

    output_noise : float, optional
        Apply noise to the output layer in the network graph, with this
        standard deviation. Defaults to 0 (no noise).

    Returns
    -------
    regs : list of :class:`Regularizer`
        A list of regularizers to apply to the given network graph.
    '''
    if 'regularizers' in kwargs:
        regs = kwargs['regularizers']
        if isinstance(regs, (tuple, list)):
            return regs
        if isinstance(regs, dict):
            kwargs.update(regs)

    regs = []

    rng = kwargs.get('rng', 13)

    def pattern(ls):
        return tuple(l.output_name for l in ls)

    inputs = pattern([l for l in graph.layers if isinstance(l, layers.Input)])
    hiddens = pattern(graph.layers[1:-1])
    outputs = pattern([graph.layers[-1]])

    # create regularizers for different types of canned dropout.
    spec = {inputs: kwargs.get('input_dropout', 0),
            hiddens: kwargs.get('hidden_dropout', 0),
            outputs: kwargs.get('output_dropout', 0)}
    spec.update(kwargs.get('dropout', {}))
    for pattern, w in spec.items():
        if w:
            regs.append(BernoulliDropout(pattern=pattern, weight=w, rng=rng))

    # create regularizers for different types of canned noise.
    spec = {inputs: kwargs.get('input_noise', 0),
            hiddens: kwargs.get('hidden_noise', 0),
            outputs: kwargs.get('output_noise', 0)}
    spec.update(kwargs.get('noise', {}))
    for pattern, w in spec.items():
        if w:
            regs.append(GaussianNoise(pattern=pattern, weight=w, rng=rng))

    # create regularizers based on other keyword arguments.
    for key, value in kwargs.items():
        if Regularizer.is_registered(key):
            if not isinstance(value, dict):
                value = dict(weight=value)
            regs.append(Regularizer.build(key, **value))

    return regs


class Regularizer(util.Registrar(str('Base'), (), {})):
    r'''A regularizer for a neural network model.

    Subclasses of this base usually either provide an implementation of the
    :func:`modify_graph` method, or an implementation of the :func:`loss` method
    (but (almost?) never both).

    Parameters
    ----------
    pattern : str
        A shell-style glob pattern describing the parameters or outputs that
        this regularizer ought to apply to.
    weight : float
        A scalar weight that indicates the "strength" of this regularizer in the
        overall loss for a model.

    Attributes
    ----------
    pattern : str
        A shell-style glob pattern describing the parameters or outputs that
        this regularizer ought to apply to.
    weight : float
        A scalar weight that indicates the "strength" of this regularizer in the
        overall loss for a model.
    '''

    def __init__(self, pattern=None, weight=0.):
        self.pattern = pattern
        self.weight = weight

    def log(self):
        '''Log some diagnostic info about this regularizer.'''
        logging.info('regularizer: %s * %s(%s)',
                     self.weight, self.__class__.__name__, self.pattern)

    def modify_graph(self, outputs):
        '''Modify the outputs of a particular layer in the computation graph.

        Parameters
        ----------
        outputs : dict of Theano expressions
            A map from string output names to the corresponding Theano
            expression. This dictionary contains the fully-scoped name of all
            outputs from a single layer in the computation graph.

            This map is mutable, so any changes that the regularizer makes will
            be retained when the caller regains control.

        Notes
        -----

        This method is applied during graph-construction time to change the
        behavior of one or more layer outputs. For example, the
        :class:`BernoulliDropout` class replaces matching outputs with an
        expression containing "masked" outputs, where some elements are randomly
        set to zero each time the expression is evaluated.

        Any regularizer that needs to modify the structure of the computation
        graph should implement this method.
        '''
        pass

    def loss(self, layers, outputs):
        '''Compute a scalar term to add to the loss function for a model.

        Parameters
        ----------
        layers : list of :class:`theanets.layers.Layer`
            A list of the layers in the model being regularized.
        outputs : dict of Theano expressions
            A dictionary mapping string expression names to their corresponding
            Theano expressions in the computation graph. This dictionary
            contains the fully-scoped name of every layer output in the graph.
        '''
        return 0.


class WeightL2(Regularizer):
    r'''Decay the weights in a model using an L2 norm penalty.

    Notes
    -----

    This regularizer implements the :func:`loss` method to add the following
    term to the network's loss function:

    .. math::
        \frac{1}{|\Omega|} \sum_{i \in \Omega} \|W_i\|_F^2

    where :math:`\Omega` is a set of "matching" weight parameters, and
    :math`\|\cdot\|_F` is the Frobenius norm (sum of squared elements).

    This regularizer tends to prevent the weights in a model from getting "too
    large." Large weights are often associated with overfitting in a model, so
    the regularizer tends to help prevent overfitting.

    Examples
    --------

    This regularizer can be specified at training or test time by providing the
    ``weight_l2`` or ``weight_decay`` keyword arguments:

    >>> net = theanets.Regression(...)

    To use this regularizer at training time:

    >>> net.train(..., weight_decay=0.1)

    By default all (2-dimensional) weights in the model are penalized. To
    include only some weights:

    >>> net.train(..., weight_decay=dict(weight=0.1, pattern='hid[23].w'))

    To use this regularizer when running the model forward to generate a
    prediction:

    >>> net.predict(..., weight_decay=0.1)

    The value associated with the keyword argument can be a scalar---in which
    case it provides the weight for the regularizer---or a dictionary, in which
    case it will be passed as keyword arguments directly to the constructor.

    References
    ----------

    .. [Moo95] J. Moody, S. Hanson, A. Krogh, & J. A. Hertz. (1995). "A simple
       weight decay can improve generalization." NIPS 4, 950-957.
    '''

    __extra_registration_keys__ = ['weight_l2', 'weight_decay']

    def loss(self, layers, outputs):
        matches = util.params_matching(layers, self.pattern or '*')
        variables = [var for _, var in matches if var.ndim > 1]
        if not variables:
            return 0
        return sum((v * v).mean() for v in variables) / len(variables)


class WeightL1(Regularizer):
    r'''Decay the weights in a model using an L1 norm penalty.

    Notes
    -----

    This regularizer implements the :func:`loss` method to add the following
    term to the network's loss function:

    .. math::
        \frac{1}{|\Omega|} \sum_{i \in \Omega} \|W_i\|_1

    where :math:`\Omega` is a set of "matching" weight parameters, and the L1
    norm :math`\|\cdot\|_1` is the sum of the absolute values of the elements in
    the matrix.

    This regularizer tends to encourage the weights in a model to be zero.
    Nonzero weights are used only when they are able to reduce the other
    components of the loss (e.g., the squared reconstruction error).

    Examples
    --------

    This regularizer can be specified at training or test time by providing the
    ``weight_l1`` or ``weight_sparsity`` keyword arguments:

    >>> net = theanets.Regression(...)

    To use this regularizer at training time:

    >>> net.train(..., weight_sparsity=0.1)

    By default all (2-dimensional) weights in the model are penalized. To
    include only some weights:

    >>> net.train(..., weight_sparsity=dict(weight=0.1, pattern='hid[23].w'))

    To use this regularizer when running the model forward to generate a
    prediction:

    >>> net.predict(..., weight_sparsity=0.1)

    The value associated with the keyword argument can be a scalar---in which
    case it provides the weight for the regularizer---or a dictionary, in which
    case it will be passed as keyword arguments directly to the constructor.

    References
    ----------

    .. [Qiu11] Q. Qiu, Z. Jiang, & R. Chellappa. (ICCV 2011). "Sparse
       dictionary-based representation and recognition of action attributes."
    '''

    __extra_registration_keys__ = ['weight_l1', 'weight_sparsity']

    def loss(self, layers, outputs):
        matches = util.params_matching(layers, self.pattern or '*')
        variables = [var for _, var in matches if var.ndim > 1]
        if not variables:
            return 0
        return sum(abs(v).mean() for v in variables) / len(variables)


class HiddenL1(Regularizer):
    r'''Penalize the activation of hidden layers under an L1 norm.

    Notes
    -----

    This regularizer implements the :func:`loss` method to add the following
    term to the network's loss function:

    .. math::
        \frac{1}{|\Omega|} \sum_{i \in \Omega} \|Z_i\|_1

    where :math:`\Omega` is a set of "matching" graph output indices, and the L1
    norm :math`\|\cdot\|_1` is the sum of the absolute values of the elements in
    the corresponding array.

    This regularizer tends to encourage the hidden unit activations in a model
    to be zero. Nonzero activations are used only when they are able to reduce
    the other components of the loss (e.g., the squared reconstruction error).

    This regularizer acts indirectly to force a model to cover the space of its
    input dataset using as few features as possible; this pressure often causes
    features to be duplicated with slight variations to "tile" the input space
    in a very different way than a non-regularized model.

    Examples
    --------

    This regularizer can be specified at training or test time by providing the
    ``hidden_l1`` or ``hidden_sparsity`` keyword arguments:

    >>> net = theanets.Regression(...)

    To use this regularizer at training time:

    >>> net.train(..., hidden_sparsity=0.1)

    By default all hidden layer outputs are penalized. To include only some
    graph outputs:

    >>> net.train(..., hidden_sparsity=dict(weight=0.1, pattern='hid3:out'))

    To use this regularizer when running the model forward to generate a
    prediction:

    >>> net.predict(..., hidden_sparsity=0.1)

    The value associated with the keyword argument can be a scalar---in which
    case it provides the weight for the regularizer---or a dictionary, in which
    case it will be passed as keyword arguments directly to the constructor.

    References
    ----------

    .. [Ng11] A. Ng. (2011). "Sparse Autoencoder." Stanford CS294A Lecture Notes
       http://web.stanford.edu/class/cs294a/sae/sparseAutoencoderNotes.pdf
    '''

    __extra_registration_keys__ = ['hidden_l1', 'hidden_sparsity']

    def loss(self, layers, outputs):
        pattern = self.pattern or [l.output_name for l in layers[1:-1]]
        matches = util.outputs_matching(outputs, pattern)
        hiddens = [expr for _, expr in matches]
        if not hiddens:
            return 0
        return sum(abs(h).mean() for h in hiddens) / len(hiddens)


class RecurrentNorm(Regularizer):
    r'''Penalize successive activation norms of recurrent layers.

    Notes
    -----

    This regularizer implements the :func:`loss` method to add the following
    term to a recurrent network's loss function:

    .. math::
        \frac{1}{T|\Omega|} \sum_{i \in \Omega} \sum_{t=1}^T
          \left( \|Z_i^t\|_2^2 - \|Z_i^{t-1}\|_2^2 \right)^2

    where :math:`\Omega` is a set of "matching" graph output indices, and the
    squared L2 norm :math`\|\cdot\|_2^2` is the sum of the squares of the
    elements in the corresponding array.

    This regularizer encourages the norms of the hidden state activations in a
    recurrent layer to remain constant over time.

    Examples
    --------

    This regularizer can be specified at training or test time by providing the
    ``recurrent_norm`` keyword argument:

    >>> net = theanets.Regression(...)

    To use this regularizer at training time:

    >>> net.train(..., recurrent_norm=dict(weight=0.1, pattern='hid3:out'))

    A ``pattern`` must be provided; this pattern will match against all outputs
    in the computation graph, so some care must be taken to ensure that the
    regularizer is applied only to specific layer outputs.

    To use this regularizer when running the model forward to generate a
    prediction:

    >>> net.predict(..., recurrent_norm=0.1)

    The value associated with the keyword argument can be a scalar---in which
    case it provides the weight for the regularizer---or a dictionary, in which
    case it will be passed as keyword arguments directly to the constructor.

    References
    ----------

    .. [Kru15] D. Krueger & R. Memisevic. (ICLR 2016?) "Regularizing RNNs by
       Stabilizing Activations." http://arxiv.org/abs/1511.08400
    '''

    __extra_registration_keys__ = ['recurrent_norm']

    def loss(self, layers, outputs):
        if self.pattern is None:
            raise util.ConfigurationError('RecurrentNorm requires a pattern!')
        matches = util.outputs_matching(outputs, self.pattern)
        hiddens = [expr for _, expr in matches]
        if not hiddens:
            return 0
        norms = ((e * e).sum(axis=-1) for e in hiddens)
        deltas = ((e[:, :-1] - e[:, 1:]).mean() for e in norms)
        return sum(deltas) / len(hiddens)


class RecurrentState(Regularizer):
    r'''Penalize state changes of recurrent layers.

    Notes
    -----

    This regularizer implements the :func:`loss` method to add the following
    term to a recurrent network's loss function:

    .. math::
        \frac{1}{T|\Omega|} \sum_{i \in \Omega} \sum_{t=1}^T
          \| Z_i^t - Z_i^{t-1} \|_2^2

    where :math:`\Omega` is a set of "matching" graph output indices, and the
    squared L2 norm :math`\|\cdot\|_2^2` is the sum of the squares of the
    elements in the corresponding array.

    This regularizer tends to encourage the hidden state activations in a
    recurrent layer to remain constant over time. Deviations from a constant
    state thus require "evidence" from the loss.

    Examples
    --------

    This regularizer can be specified at training or test time by providing the
    ``recurrent_state`` keyword argument:

    >>> net = theanets.Regression(...)

    To use this regularizer at training time:

    >>> net.train(..., recurrent_state=dict(weight=0.1, pattern='hid3:out'))

    A ``pattern`` must be provided; this pattern will match against all outputs
    in the computation graph, so some care must be taken to ensure that the
    regularizer is applied only to specific layer outputs.

    To use this regularizer when running the model forward to generate a
    prediction:

    >>> net.predict(..., recurrent_state=dict(weight=0.1, pattern='hid3:out'))

    The value associated with the keyword argument can be a scalar---in which
    case it provides the weight for the regularizer---or a dictionary, in which
    case it will be passed as keyword arguments directly to the constructor.

    References
    ----------

    '''

    __extra_registration_keys__ = ['recurrent_state']

    def loss(self, layers, outputs):
        if self.pattern is None:
            raise util.ConfigurationError('RecurrentNorm requires a pattern!')
        matches = util.outputs_matching(outputs, self.pattern)
        hiddens = [expr for _, expr in matches]
        if not hiddens:
            return 0
        deltas = (e[:, :-1] - e[:, 1:] for e in hiddens)
        return sum((d * d).mean() for d in deltas) / len(hiddens)


class Contractive(Regularizer):
    r'''Penalize the derivative of hidden layers with respect to their inputs.

    Parameters
    ----------
    wrt : str, optional
        A glob-style pattern that specifies the inputs with respect to which the
        derivative should be computed. Defaults to ``'*'``, which matches all
        inputs.

    Notes
    -----

    This regularizer implements the :func:`loss` method to add the following
    term to the network's loss function:

    .. math::
        \frac{1}{|\Omega|} \sum_{i \in \Omega} \|\frac{\partial Z_i}{x}\|_F^2

    where :math:`\Omega` is a set of "matching" graph output indices,
    :math:`Z_i` is the output of network graph :math:`i`, :math:`x` is the input
    to the network graph, and :math`\|\cdot\|_F` is the Frobenius norm (sum of
    the squared elements in the array).

    This regularizer attempts to make the derivative of the hidden representatin
    flat with respect to the input. In theory, this encourages the network to
    learn features that are insensitive to small changes in the input (that is,
    they are mostly perpindicular to the input manifold).

    Like the :class:`HiddenL1` regularizer, this acts indirectly to force a
    model to cover the space of its input dataset using as few features as
    possible; this pressure often causes features to be duplicated with slight
    variations to "tile" the input space in a very different way than a
    non-regularized model.

    Examples
    --------

    This regularizer can be specified at training or test time by providing the
    ``hidden_l1`` or ``hidden_sparsity`` keyword arguments:

    >>> net = theanets.Regression(...)

    To use this regularizer at training time:

    >>> net.train(..., contractive=0.1)

    By default all hidden layer outputs are included. To include only some graph
    outputs:

    >>> net.train(..., contractive=dict(weight=0.1, pattern='hid3:out', wrt='in'))

    To use this regularizer when running the model forward to generate a
    prediction:

    >>> net.predict(..., contractive=0.1)

    The value associated with the keyword argument can be a scalar---in which
    case it provides the weight for the regularizer---or a dictionary, in which
    case it will be passed as keyword arguments directly to the constructor.

    References
    ----------

    .. [Rif11] S. Rifai, P. Vincent, X. Muller, X. Glorot, & Y. Bengio. (ICML
       2011). "Contractive auto-encoders: Explicit invariance during feature
       extraction."

       http://machinelearning.wustl.edu/mlpapers/paper_files/ICML2011Rifai_455.pdf
    '''

    def __init__(self, pattern=None, weight=0., wrt='*'):
        self.wrt = wrt
        super(Contractive, self).__init__(pattern=pattern, weight=weight)

    def log(self):
        '''Log some diagnostic info about this regularizer.'''
        logging.info('regularizer: %s * %s(%s wrt %s)',
                     self.weight, self.__class__.__name__,
                     self.pattern, self.wrt)

    def loss(self, layer_list, outputs):
        pattern = self.pattern or [l.output_name for l in layer_list[1:-1]]
        targets = [expr for _, expr in util.outputs_matching(outputs, pattern)]
        if not targets:
            return 0
        wrt = [l.input for l in layer_list
               if isinstance(l, layers.Input) and
               fnmatch.fnmatch(l.input.name, self.wrt)]
        total = sum(TT.sqr(TT.grad(h.mean(), wrt)).mean() for h in targets)
        return total / len(targets)


class GaussianNoise(Regularizer):
    r'''Add isotropic Gaussian noise to one or more graph outputs.

    Parameters
    ----------
    rng : Theano random number generator, optional
        A Theano random number generator to use for creating noise and dropout
        values. If not provided, a new generator will be produced for this
        layer.

    Notes
    -----

    This regularizer implements the :func:`modify_graph` method to "inject"
    noise into the loss function of a network. Suppose we were optimizing a
    linear :class:`regression <theanets.feedforward.Regression>` model with one
    hidden layer under a mean squared error. The loss for an input/output pair
    :math:`(x, y)` would be:

    .. math::
        \mathcal{L} = \| V(Wx + b) + c - y \|_2^2

    where :math:`W (V)` and :math:`b (c)` are the weights and bias parameters of
    the first (and second) layers in the model.

    If we regularized this model with Gaussian noise, the loss for this pair
    would be:

    .. math::
        \mathcal{L} = \| V(W(x+\epsilon) + b) + c - y \|_2^2

    where :math:`\epsilon \sim \mathcal{N}(0, \sigma^2)` is isotropic random
    noise with standard deviation :math:`\sigma`.

    This regularizer encourages the model to develop parameter settings that are
    robust to noisy inputs. There are some parallels to the :class:`Contractive`
    regularizer, in that both models are thought to develop internal
    representations that are orthogonal to the manifold of the input data, so
    that noisy inputs are "pushed back" onto the manifold by the network.

    Examples
    --------

    This regularizer can be specified at training or test time by providing the
    ``noise`` or ``input_noise`` or ``hidden_noise`` keyword arguments:

    >>> net = theanets.Regression(...)

    To apply this regularizer at training time to network inputs:

    >>> net.train(..., input_noise=0.1)

    And to apply the regularizer to hidden states of the network:

    >>> net.train(..., hidden_noise=0.1)

    To target specific network outputs, a pattern can be given manually:

    >>> net.train(..., noise={'hid[23]:out': 0.1, 'in:out': 0.01})

    To use this regularizer when running the model forward to generate a
    prediction:

    >>> net.predict(..., input_noise=0.1)

    The value associated with the ``input_noise`` or ``hidden_noise`` keyword
    arguments should be a scalar giving the standard deviation of the noise to
    apply. The value of the ``noise`` keyword argument should be a dictionary,
    whose keys provide glob-style output name patterns, and the corresponding
    values are the noise level.

    References
    ----------

    .. [Vin10] P. Vincent, H. Larochelle, Y. Bengio, & P.-A. Manzagol. (ICML
       2008). "Extracting and composing robust features with denoising
       autoencoders."

       http://oucsace.cs.ohiou.edu/~razvan/courses/dl6900/papers/vincent08.pdf
    '''

    def __init__(self, pattern='*:out', weight=0., rng=13):
        super(GaussianNoise, self).__init__(pattern=pattern, weight=weight)
        self.rng = RandomStreams(rng) if isinstance(rng, int) else rng

    def log(self):
        '''Log some diagnostic info about this regularizer.'''
        logging.info('regularizer: %s * %s(%s)',
                     self.weight, self.__class__.__name__, self.pattern)

    def modify_graph(self, outputs):
        for name, expr in list(util.outputs_matching(outputs, self.pattern)):
            outputs[name + '-prenoise'] = expr
            outputs[name] = expr + self.rng.normal(
                size=expr.shape, std=self.weight, dtype=util.FLOAT)


class BernoulliDropout(Regularizer):
    r'''Randomly set activations of a layer output to zero.

    Parameters
    ----------

    rng : Theano random number generator, optional
        A Theano random number generator to use for creating noise and dropout
        values. If not provided, a new generator will be produced for this
        layer.

    Notes
    -----

    This regularizer implements the :func:`modify_graph` method to "inject"
    multiplicative Bernoulli noise into the loss function of a network.

    Suppose we were optimizing a linear :class:`regression
    <theanets.feedforward.Regression>` model with one hidden layer under a mean
    squared error. The loss for an input/output pair :math:`(x, y)` would be:

    .. math::
        \mathcal{L} = \| V(Wx + b) + c - y \|_2^2

    where :math:`W (V)` and :math:`b (c)` are the weights and bias parameters of
    the first (and second) layers in the model.

    If we regularized this model's input with multiplicative Bernoulli "noise,"
    the loss for this pair would be:

    .. math::
        \mathcal{L} = \| V(W(x\cdot\rho) + b) + c - y \|_2^2

    where :math:`\rho \sim \mathcal{B}(p)` is a vector of independent Bernoulli
    samples with probability :math:`p`.

    This regularizer encourages the model to develop parameter settings such
    that internal features are independent. Dropout is widely used as a powerful
    regularizer in many types of neural network models.

    Examples
    --------

    This regularizer can be specified at training or test time by providing the
    ``dropout`` or ``input_dropout`` or ``hidden_dropout`` keyword arguments:

    >>> net = theanets.Regression(...)

    To apply this regularizer at training time to network inputs:

    >>> net.train(..., input_dropout=0.1)

    And to apply the regularizer to hidden states of the network:

    >>> net.train(..., hidden_dropout=0.1)

    To target specific network outputs, a pattern can be given manually:

    >>> net.train(..., dropout={'hid[23]:out': 0.1, 'in:out': 0.01})

    To use this regularizer when running the model forward to generate a
    prediction:

    >>> net.predict(..., input_dropout=0.1)

    The value associated with the ``input_dropout`` or ``hidden_dropout``
    keyword arguments should be a scalar giving the probability of the dropout
    to apply. The value of the ``dropout`` keyword argument should be a
    dictionary, whose keys provide glob-style output name patterns, and the
    corresponding values are the dropout level.

    References
    ----------

    .. [Hin12] G. E. Hinton, N. Srivastava, A. Krizhevsky, I. Sutskever, & R. R.
       Salakhutdinov. (2012). "Improving neural networks by preventing
       co-adaptation of feature detectors." http://arxiv.org/pdf/1207.0580.pdf
    '''

    def __init__(self, pattern='*:out', weight=0., rng=13):
        super(BernoulliDropout, self).__init__(pattern=pattern, weight=weight)
        self.rng = RandomStreams(rng) if isinstance(rng, int) else rng

    def log(self):
        '''Log some diagnostic info about this regularizer.'''
        logging.info('regularizer: %s(%s) ~ B(%s)',
                     self.__class__.__name__, self.pattern, self.weight)

    def modify_graph(self, outputs):
        for name, expr in list(util.outputs_matching(outputs, self.pattern)):
            noise = self.rng.binomial(
                size=expr.shape, n=1, p=1-self.weight, dtype=util.FLOAT)
            outputs[name + '-predrop'] = expr
            outputs[name] = expr * noise / (1-self.weight)

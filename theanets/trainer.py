# -*- coding: utf-8 -*-

'''This module contains optimization methods for neural networks.

Most of the optimization methods (mostly the ones based on gradient descent) are
general-purpose optimization routines that happen to be pretty good for training
neural networks. Other methods --- :class:`Sample`,
:class:`SupervisedPretrainer`, and :class:`UnsupervisedPretrainer` --- are
specific to neural networks. Despite the difference in generality, all of the
training routines implemented here assume that a :class:`Network
<theanets.feedforward.Network>` is being optimized.

Most of the general-purpose optimization routines in this module are based on
the :class:`SGD` parent and optimize the loss function at hand by taking small
steps in the general direction of the local gradient of the loss. Such
stochastic gradient optimization techniques are not bad, in the sense that they
will generally always take steps that reduce the loss, but because they use
local gradient information, they are not guaranteed to find a global optimum for
nonlinear losses. Whether this is a problem or not depends on your task, but
these approaches have been shown to be quite useful in the past couple decades
of machine learning research.
'''

import climate
import collections
import itertools
import numpy as np
import numpy.random as rng
import scipy.optimize
import theano
import theano.tensor as TT
import sys

from . import feedforward
from . import layers

logging = climate.get_logger(__name__)

FLOAT = theano.config.floatX


def default_mapper(f, dataset, *args, **kwargs):
    '''Apply (map) a function to each element of a dataset.'''
    return [f(x, *args, **kwargs) for x in dataset]


def ipcluster_mapper(client):
    '''Get a mapper from an IPython.parallel cluster client.

    This helper is experimental and not currently used.

    Parameters
    ----------
    client : :ipy:`IPython.parallel.Client`
        A client for an IPython cluster. The dataset will be processed by
        distributing it across the cluster.

    Returns
    -------
    mapper : callable
        A callable that can be used to map a dataset to a function across an
        IPython cluster.
    '''
    view = client.load_balanced_view()
    def mapper(f, dataset, *args, **kwargs):
        def ff(x):
            return f(x, *args, **kwargs)
        return view.map(ff, dataset).get()
    return mapper


class Trainer(object):
    '''All trainers derive from this base class.

    A trainer is a wrapper for a few different types of theano functions, along
    with the parameters that define the behavior of these functions. The trainer
    base class is abstract; subclasses must provide an implementation of the
    :func:`train` method.

    Attributes
    ----------
    params : list of theano variables
        Parameters from our network model that require training.
    loss : theano expression
        An expression for computing a scalar loss value for the network, given
        the current parameter settings.
    f_eval : theano function
        A function that takes some data and returns a sequence of monitor values
        for that data.

    Parameters
    ----------
    validate_every : int, optional
        Validate the model after this many training iterations have passed.
        Defaults to 10.
    min_improvement : float, optional
        Quit training if the evaluation loss for the model does not improve by
        at least this relative amount for `patience` validations. Defaults to 0,
        meaning that any improvement to the validation loss counts.
    patience : int, optional
        Maximum number of validations that can pass before the validation loss
        must improve by `min_improvement` relative. Defaults to 10.
    '''

    def __init__(self, network, **kwargs):
        super(Trainer, self).__init__()

        self.validate_every = kwargs.get('validate_every', 10)
        self.min_improvement = kwargs.get('min_improvement', 0.)
        self.patience = kwargs.get('patience', 10)

        self.params = network.params
        self._shapes = [p.get_value(borrow=True).shape for p in self.params]
        self._counts = [np.prod(s) for s in self._shapes]
        self._starts = np.cumsum([0] + self._counts)[:-1]
        self._dtype = self.params[0].get_value().dtype

        self._best_loss = 1e100
        self._best_iter = self._curr_iter = 0
        self._best_params = [p.get_value().copy() for p in self.params]

        self.loss = network.loss(**kwargs)
        self._monitor_exprs = [self.loss]
        self._monitor_names = ['loss']
        for name, monitor in network.monitors:
            self._monitor_names.append(name)
            self._monitor_exprs.append(monitor)

        logging.info('compiling evaluation function')
        self.f_eval = theano.function(
            network.inputs, self._monitor_exprs, updates=network.updates)

    def set_params(self, targets):
        '''Set the values of the parameters to the given target values.

        Parameters
        ----------
        targets : sequence of ndarray
            Arrays for setting the parameters of our model.
        '''
        for param, target in zip(self.params, targets):
            param.set_value(target)

    def log(self, monitors, iteration, label='', suffix=''):
        '''Log the state of the model through the logging system.

        Parameters
        ----------
        monitors : OrderedDict
            A dictionary of monitor names mapped to values. These names and
            values are what is being logged.
        iteration : int
            Training iteration that we are logging.
        label : str, optional
            A label for the name of the trainer creating the log line. Defaults
            to the name of the current class.
        suffix : str, optional
            A suffix to add to the end of the log line, if any.
        '''
        label = label or self.__class__.__name__
        fields = []
        for name, value in monitors.items():
            width = '{:.2f}'
            if name == 'loss':
                width = '{:.6f}'
            elif '<' in name or '>' in name:
                width = '{:.1f}'
            fields.append(('{}=' + width).format(name, value))
        logging.info('%s %i %s%s', label, iteration, ' '.join(fields), suffix)

    def evaluate(self, dataset):
        '''Evaluate the current model parameters on a dataset.

        Parameters
        ----------
        dataset : :class:`theanets.dataset.Dataset`
            A set of data to use for evaluating the model.

        Returns
        -------
        monitors : OrderedDict
            A dictionary mapping monitor names to values. Monitors are
            quantities of interest during training---for example, loss function,
            accuracy, or whatever the layers in the network define.
        '''
        values = [self.f_eval(*x) for x in dataset]
        monitors = zip(self._monitor_names, np.mean(values, axis=0))
        return collections.OrderedDict(monitors)

    def test_patience(self, monitors):
        '''Test whether our patience with training has elapsed.

        Parameters
        ----------
        monitors : dict
            A dictionary mapping monitor names to values. The 'loss' key from
            this dictionary will be used to evaluate training progress.

        Returns
        -------
        elapsed : bool
            True iff our patience has elapsed and the model is no longer
            improving.
        '''
        self._curr_iter += 1
        marker = ''
        loss = monitors['loss']
        if self._best_loss - loss > self._best_loss * self.min_improvement:
            self._best_loss = loss
            self._best_iter = self._curr_iter
            self._best_params = [p.get_value().copy() for p in self.params]
            marker = ' *'
        self.log(monitors, self._curr_iter - 1, 'validation', marker)
        return self._curr_iter - self._best_iter > self.patience

    def itertrain(self, train_set, valid_set=None, **kwargs):
        '''Train a model using a training and validation set.

        This method yields a series of monitor values to the caller. After every
        iteration, a pair of monitor dictionaries is generated: one evaluated on
        the training dataset, and another evaluated on the validation dataset.
        The validation monitors might not be updated during every training
        iteration; in this case, the most recent validation monitors will be
        yielded along with the training monitors.

        Parameters
        ----------
        train_set : :class:`theanets.dataset.Dataset`
            A set of training data for computing updates to model parameters.
        valid_set : :class:`theanets.dataset.Dataset`
            A set of validation data for computing monitor values and
            determining when the loss has stopped improving.

        Returns
        -------
        training : dict
            A dictionary mapping monitor names to values, evaluated on the
            training dataset.
        validation : dict
            A dictionary containing monitor values evaluated on the validation
            dataset.
        '''
        iteration = 0
        training = validation = None
        while True:
            if not iteration % self.validate_every:
                try:
                    validation = self.evaluate(valid_set)
                except KeyboardInterrupt:
                    logging.info('interrupted!')
                    break
                if self.test_patience(validation):
                    logging.info('patience elapsed!')
                    break
            try:
                training = self.step(train_set)
            except KeyboardInterrupt:
                logging.info('interrupted!')
                break
            iteration += 1
            self.log(training, iteration)
            yield training, validation
        self.set_params(self._best_params)


class SGD(Trainer):
    r'''Optimize using stochastic gradient descent with momentum.

    A stochastic gradient trainer with momentum :math:`\mu` and learning rate
    :math:`\alpha` updates parameter :math:`p` at step :math:`t` by blending the
    current "velocity" :math:`v` with the current gradient
    :math:`\frac{\partial\mathcal{L}}{\partial p}`:

    .. math::
        \begin{eqnarray*}
        v_{t+1} &=& \mu v_t - \alpha \frac{\partial\mathcal{L}}{\partial p} \\
        p_{t+1} &=& p_t + v_{t+1}
        \end{eqnarray*}

    Without momentum (or when :math:`\mu = 0`), these updates reduce to
    :math:`p_{t+1} = p_t - \alpha \frac{\partial\mathcal{L}}{\partial p}`, which
    just takes steps downhill according to the the local gradient. Adding the
    momentum term permits the algorithm to incorporate information from previous
    steps as well, which in practice has the effect of incorporating some
    information about second-order derivatives of the loss surface.
    '''

    def __init__(self, network, **kwargs):
        super(SGD, self).__init__(network, **kwargs)

        self.clip = TT.cast(kwargs.get('gradient_clip', 1e6), FLOAT)
        self.max_norm = TT.cast(kwargs.get('max_gradient_norm', 1e6), FLOAT)
        self.momentum = TT.cast(kwargs.get('momentum', 0.9), FLOAT)
        self.learning_rate = TT.cast(kwargs.get('learning_rate', 1e-4), FLOAT)

        logging.info('compiling %s learning function', self.__class__.__name__)
        updates = list(network.updates) + list(self.learning_updates())
        self.f_learn = theano.function(
            network.inputs, self._monitor_exprs, updates=updates)

    def learning_updates(self):
        for param, grad in zip(self.params, self.clipped_gradients()):
            vel_tm1 = self.shared_like(param, 'vel')
            vel_t = self.momentum * vel_tm1 - self.learning_rate * grad
            yield vel_tm1, vel_t
            yield param, param + vel_t

    def clipped_gradients(self, params=None):
        for grad in TT.grad(self.loss, params or self.params):
            clip = TT.clip(grad, -self.clip, self.clip)
            norm = TT.sqrt((grad * grad).sum())
            yield clip * TT.minimum(1, self.max_norm / norm)

    @staticmethod
    def shared_like(param, name, init=0):
        return theano.shared(np.zeros_like(param.get_value()) + init,
                             name='{}_{}'.format(param.name, name))

    def step(self, dataset):
        '''Advance the state of the model by one training step.

        Parameters
        ----------
        dataset : :class:`Dataset <theanets.dataset.Dataset>`
            A dataset for training the model.

        Returns
        -------
        training : dict
            A dictionary mapping monitor names to values.
        '''
        values = [self.f_learn(*x) for x in dataset]
        return collections.OrderedDict(
            zip(self._monitor_names, np.mean(values, axis=0)))


class NAG(SGD):
    r'''Optimize using Nesterov's Accelerated Gradient (NAG).

    The basic difference between NAG and "classical" momentum in SGD
    optimization approaches is that NAG computes the gradients at the position
    in parameter space where "classical" momentum would put us at the *next*
    step. In classical :class:`SGD` with momentum :math:`\mu` and learning rate
    :math:`\alpha`, updates to parameter :math:`p` at step :math:`t` are
    computed by blending the current "velocity" :math:`v` with the current
    gradient :math:`\frac{\partial\mathcal{L}}{\partial p}`:

    .. math::
        \begin{eqnarray*}
        v_{t+1} &=& \mu v_t - \alpha \frac{\partial\mathcal{L}}{\partial p} \\
        p_{t+1} &=& p_t + v_{t+1}
        \end{eqnarray*}

    In contrast, NAG adjusts the update by blending the current "velocity" with
    the gradient at the next step---that is, the gradient is computed at the
    point where the velocity would have taken us:

    .. math::
        \begin{eqnarray*}
        v_{t+1} &=& \mu v_t - \alpha \left.\frac{\partial\mathcal{L}}{\partial p}\right|_{p_t + \mu v_t} \\
        p_{t+1} &=& p_t + v_{t+1}
        \end{eqnarray*}

    Again, the difference here is that the gradient is computed at the place in
    parameter space where we would have stepped using the classical technique,
    in the absence of a new gradient.

    In theory, this helps correct for oversteps during learning: If momentum
    would lead us to overshoot, then the gradient at that overshot place will
    point backwards, toward where we came from. For details on this idea, see
    Sutskever, Martens, Dahl, and Hinton, "On the importance of initialization
    and momentum in deep learning" (ICML 2013)
    http://jmlr.csail.mit.edu/proceedings/papers/v28/sutskever13.pdf
    '''

    def learning_updates(self):
        # see https://github.com/lisa-lab/pylearn2/pull/136#issuecomment-10381617
        for param, grad in zip(self.params, self.clipped_gradients()):
            vel_tm1 = self.shared_like(param, 'vel')
            vel_t = self.momentum * vel_tm1 - self.learning_rate * grad
            yield vel_tm1, vel_t
            yield param, param + self.momentum * vel_t - self.learning_rate * grad


class Rprop(SGD):
    r'''Trainer for neural nets using resilient backpropagation.

    The Rprop method uses the same general strategy as SGD (both methods are
    make small parameter adjustments using local derivative information). The
    difference is that in Rprop, only the signs of the partial derivatives are
    taken into account when making parameter updates. That is, the step size for
    each parameter is independent of the magnitude of the gradient for that
    parameter.

    To accomplish this, Rprop maintains a separate learning rate for every
    parameter in the model, and adjusts this learning rate based on the
    consistency of the sign of the gradient of the loss with respect to that
    parameter over time. Whenever two consecutive gradients for a parameter have
    the same sign, the learning rate for that parameter increases, and whenever
    the signs disagree, the learning rate decreases. This has a similar effect
    to momentum-based SGD methods but effectively maintains parameter-specific
    learning rates.

    .. math::
        \begin{eqnarray*}
        && \mbox{if } \frac{\partial\mathcal{L}}{\partial p}_{t-1}\frac{\partial\mathcal{L}}{\partial p} > 0 \\
        && \qquad \Delta_t = \min (\eta_+\Delta_{t−1}, \Delta_+) \\
        && \mbox{if } \frac{\partial\mathcal{L}}{\partial p}_{t-1}\frac{\partial\mathcal{L}}{\partial p} < 0 \\
        && \qquad \Delta_t = \max (\eta_-\Delta_{t−1}, \Delta_-) \\
        && \qquad \frac{\partial\mathcal{L}}{\partial p} = 0 \\
        && p_{t+1} = p_t − \mbox{sgn}\left(\frac{\partial\mathcal{L}}{\partial p}\right) \Delta_t
        \end{eqnarray*}

    Here, :math:`s(\cdot)` is the sign function (i.e., returns -1 if its
    argument is negative and 1 otherwise), :math:`\eta_-` and :math:`\eta_+` are
    the amount to decrease (increase) the step size if the gradients disagree
    (agree) in sign, and :math:`\Delta_+` and :math:`\Delta_-` are the maximum
    and minimum step size.

    The implementation here is actually the "iRprop-" variant of Rprop described
    in Algorithm 4 from Igel and Huesken, "Improving the Rprop Learning
    Algorithm" (2000). This variant resets the running gradient estimates to
    zero in cases where the previous and current gradients have switched signs.
    '''

    def __init__(self, network, **kwargs):
        self.step_increase = TT.cast(kwargs.get('rprop_increase', 1.01), FLOAT)
        self.step_decrease = TT.cast(kwargs.get('rprop_decrease', 0.99), FLOAT)
        self.min_step = TT.cast(kwargs.get('rprop_min_step', 0.), FLOAT)
        self.max_step = TT.cast(kwargs.get('rprop_max_step', 100.), FLOAT)
        super(Rprop, self).__init__(network, **kwargs)

    def learning_updates(self):
        for param, grad in zip(self.params, self.clipped_gradients()):
            grad_tm1 = self.shared_like(param, 'grad')
            step_tm1 = self.shared_like(param, 'step', self.learning_rate.value)
            test = grad * grad_tm1
            same = TT.gt(test, 0)
            diff = TT.lt(test, 0)
            step = TT.minimum(self.max_step, TT.maximum(self.min_step, step_tm1 * (
                TT.eq(test, 0) +
                same * self.step_increase +
                diff * self.step_decrease)))
            grad = grad - diff * grad
            yield param, param - TT.sgn(grad) * step
            yield grad_tm1, grad
            yield step_tm1, step


class RmsProp(SGD):
    r'''RmsProp trains neural network models using scaled SGD.

    The RmsProp method uses the same general strategy as SGD, in the sense that
    all gradient-based methods make small parameter adjustments using local
    derivative information. The difference here is that as gradients are
    computed during each parameter update, an exponential moving average of
    gradient magnitudes is maintained as well. At each update, the EMA is used
    to compute the root-mean-square (RMS) gradient value that's been seen in the
    recent past. The actual gradient is normalized by this RMS scaling factor
    before being applied to update the parameters.

    .. math::
        \begin{eqnarray*}
        f_{t+1} &=& \gamma a_t + (1 - \gamma) \frac{\partial\mathcal{L}}{\partial p} \\
        g_{t+1} &=& \gamma g_t + (1 - \gamma) \left(\frac{\partial\mathcal{L}}{\partial p}\right)^2 \\
        v_{t+1} &=& \mu v_t - \frac{\alpha}{\sqrt{g_{t+1} - f_{t+1}^2 + \epsilon}} \frac{\partial\mathcal{L}}{\partial p} \\
        p_{t+1} &=& p_t + v_{t+1}
        \end{eqnarray*}

    Like Rprop, this learning method effectively maintains a sort of
    parameter-specific momentum value, but this method takes into account both
    the sign and the magnitude of the gradient for each parameter.

    In this implementation, :math:`\epsilon = 0.0001`, and the weight parameter
    :math:`\gamma` for the EMA window is computed from the ``rms_halflife``
    keyword argument, such that the actual EMA weight varies inversely with the
    halflife :math:`h`: :math:`\gamma = e^{\frac{-\ln 2}{h}}`.

    The implementation here is taken from Graves, "Generating Sequences With
    Recurrent Neural Networks" (2013), equations (38)--(45); the paper is
    available at http://arxiv.org/abs/1308.0850. Graves' implementation in
    particular seems to have introduced the :math:`f_t` terms into the RMS
    computation; these terms appear to act as a sort of momentum for the RMS
    values.
    '''

    def __init__(self, network, **kwargs):
        self.ewma = TT.cast(np.exp(-np.log(2) / kwargs.get('rms_halflife', 7)), FLOAT)
        super(RmsProp, self).__init__(network, **kwargs)

    def learning_updates(self):
        for param, grad in zip(self.params, self.clipped_gradients()):
            g1_tm1 = self.shared_like(param, 'g1_ewma')
            g2_tm1 = self.shared_like(param, 'g2_ewma')
            vel_tm1 = self.shared_like(param, 'vel')
            g1_t = self.ewma * g1_tm1 + (1 - self.ewma) * grad
            g2_t = self.ewma * g2_tm1 + (1 - self.ewma) * grad * grad
            rms = TT.sqrt(g2_t - g1_t * g1_t + 1e-4)
            vel_t = self.momentum * vel_tm1 - grad * self.learning_rate / rms
            yield g1_tm1, g1_t
            yield g2_tm1, g2_t
            yield vel_tm1, vel_t
            yield param, param + vel_t


class ADADELTA(RmsProp):
    r'''ADADELTA trains neural network models using scaled :class:`SGD`.

    The ADADELTA method uses the same general strategy as :class:`SGD` (both
    methods are make small parameter adjustments using local derivative
    information). The difference here is that as gradients are computed during
    each parameter update, an exponential weighted moving average gradient
    value, as well as an exponential weighted moving average of recent parameter
    steps, are maintained as well. The actual gradient is normalized by the
    ratio of the parameter step RMS values to the gradient RMS values.

    .. math::
        \begin{eqnarray*}
        g_{t+1} &=& \gamma g_t + (1 - \gamma) \left(\frac{\partial\mathcal{L}}{\partial p}\right)^2 \\
        v_{t+1} &=& -\frac{\sqrt{x_t + \epsilon}}{\sqrt{g_{t+1} + \epsilon}} \frac{\partial\mathcal{L}}{\partial p} \\
        x_{t+1} &=& \gamma x_t + (1 - \gamma) v_{t+1}^2 \\
        p_{t+1} &=& p_t + v_{t+1}
        \end{eqnarray*}

    Like :class:`Rprop` and :class:`RmsProp`, this learning method effectively
    maintains a sort of parameter-specific momentum value. The primary
    difference between this method and :class:`RmsProp` is that ADADELTA
    additionally incorporates a sliding window of RMS parameter steps, obviating
    the need for a learning rate parameter.

    In this implementation, :math:`\epsilon` is taken from the ``learning_rate``
    keyword argument. The weight parameter :math:`\gamma` for the EMA window is
    computed from the ``rms_halflife`` keyword argument, such that the actual
    EMA weight varies inversely with the halflife :math:`h`: :math:`\gamma =
    e^{\frac{-\ln 2}{h}}`.

    The implementation is modeled after Zeiler (2012), "ADADELTA: An adaptive
    learning rate method," available at http://arxiv.org/abs/1212.5701.
    '''

    def learning_updates(self):
        eps = self.learning_rate
        for param, grad in zip(self.params, self.clipped_gradients()):
            x2_tm1 = self.shared_like(param, 'x2_ewma')
            g2_tm1 = self.shared_like(param, 'g2_ewma')
            g2_t = self.ewma * g2_tm1 + (1 - self.ewma) * grad * grad
            delta = grad * TT.sqrt(x2_tm1 + eps) / TT.sqrt(g2_t + eps)
            x2_t = self.ewma * x2_tm1 + (1 - self.ewma) * delta * delta
            yield g2_tm1, g2_t
            yield x2_tm1, x2_t
            yield param, param - delta


class Scipy(Trainer):
    '''General trainer for neural nets using ``scipy``.

    This class serves as a wrapper for the optimization algorithms provided in
    `scipy.optimize.minimize`_. The following algorithms are available in this
    trainer:

    - ``bfgs``
    - ``cg``
    - ``dogleg``
    - ``newton-cg``
    - ``trust-ncg``

    In general, these methods require two types of computations in order to
    minimize a cost function: evaluating the cost function for a specific
    setting of model parameters, and computing the gradient of the cost function
    for a specific setting of model parameters. Both of these computations are
    implemented by the ``theanets`` package and may, if you have a GPU, involve
    computing values on the GPU.

    However, all of the optimization steps that might be performed once these
    two types of values are computed will not be handled on the GPU, since
    ``scipy`` is not capable of using the GPU. This might or might not influence
    the absolute time required to optimize a model, depending on the ratio of
    time spent computing cost and gradient values to the time spent computing
    parameter updates.

    For more information about these optimization methods, please see the `Scipy
    documentation`_.

    .. _scipy.optimize.minimize: http://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html
    .. _Scipy documentation: http://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html
    '''

    METHODS = ('bfgs', 'cg', 'dogleg', 'newton-cg', 'trust-ncg')

    def __init__(self, network, method, **kwargs):
        super(Scipy, self).__init__(network, **kwargs)

        self.method = method

        logging.info('compiling gradient function')
        self.f_grad = theano.function(network.inputs, TT.grad(self.loss, self.params))

    def flat_to_arrays(self, x):
        '''Convert a parameter vector to a sequence of parameter arrays.

        Parameters
        ----------
        flat : ndarray
            A one-dimensional numpy array containing flattened parameter values
            for all parameters in our model.

        Returns
        -------
        arrays : sequence of ndarray
            Values of the parameters in our model.
        '''
        x = x.astype(self._dtype)
        return [x[o:o+n].reshape(s) for s, o, n in
                zip(self._shapes, self._starts, self._counts)]

    def arrays_to_flat(self, arrays):
        '''Convert a sequence of parameter arrays to a vector.

        Parameters
        ----------
        arrays : sequence of ndarray
            Values of the parameters in our model.

        Returns
        -------
        flat : ndarray
            A one-dimensional numpy array containing flattened parameter values
            for all parameters in our model.
        '''
        x = np.zeros((sum(self._counts), ), self._dtype)
        for arr, o, n in zip(arrays, self._starts, self._counts):
            x[o:o+n] = arr.ravel()
        return x

    def function_at(self, x, dataset):
        '''Compute the value of the loss function at given parameter values.

        Parameters
        ----------
        x : ndarray
            An array of parameter values to set our model at.
        dataset : :class:`theanets.dataset.Dataset`
            A set of data over which to compute our loss function.

        Returns
        -------
        loss : float
            Scalar value of the loss function, evaluated at the given parameter
            settings, using the given dataset.
        '''
        self.set_params(self.flat_to_arrays(x))
        return self.evaluate(dataset)['loss']

    def gradient_at(self, x, dataset):
        '''Compute the gradients of the loss function at given parameter values.

        Parameters
        ----------
        x : ndarray
            An array of parameter values to set our model at.
        dataset : :class:`theanets.dataset.Dataset`
            A set of data over which to compute our gradients.

        Returns
        -------
        gradients : ndarray
            A vector of gradient values, of the same dimensions as `x`.
        '''
        self.set_params(self.flat_to_arrays(x))
        grads = [[] for _ in range(len(self.params))]
        for x in dataset:
            for i, g in enumerate(self.f_grad(*x)):
                grads[i].append(np.asarray(g))
        return self.arrays_to_flat([np.mean(g, axis=0) for g in grads])

    def step(self, dataset):
        '''Advance the state of the model by one training step.

        Parameters
        ----------
        dataset : :class:`Dataset <theanets.dataset.Dataset>`
            A dataset for training the model.

        Returns
        -------
        training : dict
            A dictionary mapping monitor names to values.
        '''
        res = scipy.optimize.minimize(
            fun=self.function_at,
            jac=self.gradient_at,
            x0=self.arrays_to_flat(self._best_params),
            args=(dataset, ),
            method=self.method,
            options=dict(maxiter=self.validate_every),
        )
        self.set_params(self.flat_to_arrays(res.x))
        return self.evaluate(dataset)


class LM(Trainer):
    '''Levenberg-Marquardt trainer for neural networks.

    Based on the description of the algorithm in "Levenberg-Marquardt
    Optimization" by Sam Roweis.
    '''

    def __init__(self, network, **kwargs):
        raise NotImplementedError


class HF(Trainer):
    '''The hessian free trainer shells out to an external implementation.

    hf.py was implemented by Nicholas Boulanger-Lewandowski and made available
    to the public (yay !). If you don't have a copy of the module handy, this
    class will attempt to download it from github.
    '''

    URL = 'https://raw.github.com/boulanni/theano-hf/master/hf.py'

    def __init__(self, network, **kwargs):
        import os, tempfile
        try:
            import urllib.request
        except: # Python 2.x
            import urllib
        sys.path.append(tempfile.gettempdir())

        try:
            import hf
        except:
            # if hf failed to import, try downloading it and saving it locally.
            logging.error('hf import failed, attempting to download %s', HF.URL)
            path = os.path.join(tempfile.gettempdir(), 'hf.py')
            try:
                urllib.request.urlretrieve(HF.URL, path)
            except: # Python 2.x
                urllib.urlretrieve(HF.URL, path)
            logging.info('downloaded hf code to %s', path)
            import hf

        self.params = network.params
        self.opt = hf.hf_optimizer(
            self.params,
            network.inputs,
            network.outputs[0],
            [network.loss(**kwargs)] + [mon for _, mon in network.monitors],
            None)

        # fix mapping from kwargs into a dict to send to the hf optimizer
        kwargs['validate_every'] = kwargs.pop('validate', 1 << 60)
        try:
            func = self.opt.train.__func__.__code__
        except: # Python 2.x
            func = self.opt.train.im_func.func_code
        for k in set(kwargs) - set(func.co_varnames[1:]):
            kwargs.pop(k)
        self.kwargs = kwargs

    def itertrain(self, train_set, valid_set=None, **kwargs):
        '''Train a model using a training and validation set.

        This method yields a series of monitor values to the caller. After every
        iteration, a pair of monitor dictionaries is generated: one evaluated on
        the training dataset, and another evaluated on the validation dataset.
        The validation monitors might not be updated during every training
        iteration; in this case, the most recent validation monitors will be
        yielded along with the training monitors.

        Parameters
        ----------
        train_set : :class:`theanets.dataset.Dataset`
            A set of training data for computing updates to model parameters.
        valid_set : :class:`theanets.dataset.Dataset`
            A set of validation data for computing monitor values and
            determining when the loss has stopped improving.

        Returns
        -------
        training : dict
            A dictionary mapping monitor names to values, evaluated on the
            training dataset.
        validation : dict
            A dictionary containing monitor values evaluated on the validation
            dataset.
        '''
        self.set_params(self.opt.train(
            train_set, kwargs['cg_set'], validation=valid_set, **self.kwargs))
        yield self.evaluate(train_set), self.evaluate(valid_set)


class Sample(Trainer):
    '''This trainer replaces network weights with samples from the input.'''

    @staticmethod
    def reservoir(xs, n):
        '''Select a random sample of n items from xs.'''
        pool = []
        for i, x in enumerate(xs):
            if len(pool) < n:
                pool.append(x / np.linalg.norm(x))
                continue
            j = rng.randint(i + 1)
            if j < n:
                pool[j] = x / np.linalg.norm(x)
        # if the pool still has fewer than n items, pad with distorted random
        # duplicates from the source data.
        L = len(pool)
        S = np.std(pool, axis=0)
        while len(pool) < n:
            x = pool[rng.randint(L)]
            pool.append(x + S * rng.randn(*x.shape))
        return np.array(pool, dtype=pool[0].dtype)

    def __init__(self, network, **kwargs):
        self.network = network

    def itertrain(self, train_set, valid_set=None, **kwargs):
        '''Train a model using a training and validation set.

        This method yields a series of monitor values to the caller. After every
        iteration, a pair of monitor dictionaries is generated: one evaluated on
        the training dataset, and another evaluated on the validation dataset.
        The validation monitors might not be updated during every training
        iteration; in this case, the most recent validation monitors will be
        yielded along with the training monitors.

        Parameters
        ----------
        train_set : :class:`theanets.dataset.Dataset`
            A set of training data for computing updates to model parameters.
        valid_set : :class:`theanets.dataset.Dataset`
            A set of validation data for computing monitor values and
            determining when the loss has stopped improving.

        Returns
        -------
        training : dict
            A dictionary mapping monitor names to values, evaluated on the
            training dataset.
        validation : dict
            A dictionary containing monitor values evaluated on the validation
            dataset.
        '''
        ifci = itertools.chain.from_iterable

        first = lambda x: x[0] if isinstance(x, (tuple, list)) else x
        last = lambda x: x[-1] if isinstance(x, (tuple, list)) else x
        odim = idim = None
        for t in train_set:
            idim = first(t).shape[-1]
            odim = last(t).shape[-1]

        # set output (decoding) weights on the network.
        samples = ifci(last(t) for t in train_set)
        for param in self.network.layers[-1].params:
            shape = param.get_value(borrow=True).shape
            if len(shape) == 2 and shape[1] == odim:
                arr = np.vstack(Sample.reservoir(samples, shape[0]))
                logging.info('setting %s: %s <- %s', param.name, shape)
                param.set_value(arr / np.sqrt((arr * arr).sum(axis=1))[:, None])

        # set input (encoding) weights on the network.
        samples = ifci(first(t) for t in train_set)
        for layer in self.network.layers:
            for param in layer.params:
                shape = param.get_value(borrow=True).shape
                if len(shape) == 2 and shape[0] == idim:
                    arr = np.vstack(Sample.reservoir(samples, shape[1])).T
                    logging.info('setting %s: %s', param.name, shape)
                    param.set_value(arr / np.sqrt((arr * arr).sum(axis=0)))
                    samples = ifci(self.network.feed_forward(
                        first(t))[i-1] for t in train_set)

        yield self.evaluate(train_set), self.evaluate(valid_set)


class SupervisedPretrainer(Trainer):
    '''This trainer adapts parameters using a supervised pretraining approach.

    In this variant, we create "taps" at increasing depths into the original
    network weights, training only those weights that are below the tap. So, for
    a hypothetical binary classifier network with layers [3, 4, 5, 6, 2], we
    would first insert a tap after the first hidden layer (effectively a binary
    classifier in a [3, 4, (2)] configuration, where (2) indicates that the
    corresponding layer is the tap, not present in the original) and train just
    that network. Then we insert a tap at the next layer (effectively training a
    [3, 4, 5, (2)] classifier, re-using the trained weights for the 3 x 4
    layer), and so forth. When we get to training the last layer, i.e., [3, 4,
    5, 6, 2], then we just train all of the layers in the original network.

    For autoencoder networks with tied weights, consider an example with layers
    [3, 4, 5, 6, 5', 4', 3'], where the prime indicates that the layer is tied.
    In cases like this, we train the "outermost" pair of layers first, then add
    then next pair of layers inward, etc. The training for our example would
    start with [3, 4, 3'], then proceed to [3, 4, 5, 4', 3'], and then finish by
    training all the layers in the original network.

    By using layers from the original network whenever possible, we preserve all
    of the relevant settings of noise, dropouts, loss function and the like, in
    addition to removing the need for copying trained weights around between
    different :class:`theanets.feedforward.Network` instances.
    '''

    def __init__(self, network, factory, *args, **kwargs):
        self.network = network
        self.factory = factory
        self.args = args
        self.kwargs = kwargs

    def itertrain(self, train_set, valid_set=None, **kwargs):
        '''Train a model using a training and validation set.

        This method yields a series of monitor values to the caller. After every
        iteration, a pair of monitor dictionaries is generated: one evaluated on
        the training dataset, and another evaluated on the validation dataset.
        The validation monitors might not be updated during every training
        iteration; in this case, the most recent validation monitors will be
        yielded along with the training monitors.

        Parameters
        ----------
        train_set : :class:`theanets.dataset.Dataset`
            A set of training data for computing updates to model parameters.
        valid_set : :class:`theanets.dataset.Dataset`
            A set of validation data for computing monitor values and
            determining when the loss has stopped improving.

        Returns
        -------
        training : dict
            A dictionary mapping monitor names to values, evaluated on the
            training dataset.
        validation : dict
            A dictionary containing monitor values evaluated on the validation
            dataset.
        '''
        net = self.network
        tied = getattr(net, 'tied_weights', False)
        original = list(net.layers)
        L = 1 + len(original) // 2 if tied else len(original) - 1
        for i in range(1, L):
            if i == L - 1:
                net.layers = original
            elif tied:
                net.layers = original[:i+1] + original[-i:]
            else:
                net.layers = original[:i+1] + [layers.build(
                    'feedforward',
                    name='lwout',
                    nin=original[i].nout,
                    nout=original[-1].nout,
                    activation=original[-1].kwargs['activation'])]
            logging.info('layerwise: training %s',
                         ' -> '.join(l.name for l in net.layers))
            trainer = self.factory(net, *self.args, **self.kwargs)
            for monitors in trainer.itertrain(train_set, valid_set):
                yield monitors
        net.layers = original


class UnsupervisedPretrainer(Trainer):
    '''Train a discriminative model using an unsupervised pre-training step.

    This trainer is a bit of glue code that creates a "shadow" autoencoder based
    on a current network model, trains the autoencoder, and then transfers the
    trained weights back to the original model.

    This code is intended mostly as a proof-of-concept to demonstrate how shadow
    networks can be created, and how trainers can call other trainers for lots
    of different types of training regimens.
    '''

    def __init__(self, network, *args, **kwargs):
        self.network = network
        self.args = args
        self.kwargs = kwargs

    def itertrain(self, train_set, valid_set=None, **kwargs):
        '''Train a model using a training and validation set.

        This method yields a series of monitor values to the caller. After every
        iteration, a pair of monitor dictionaries is generated: one evaluated on
        the training dataset, and another evaluated on the validation dataset.
        The validation monitors might not be updated during every training
        iteration; in this case, the most recent validation monitors will be
        yielded along with the training monitors.

        Parameters
        ----------
        train_set : :class:`theanets.dataset.Dataset`
            A set of training data for computing updates to model parameters.
        valid_set : :class:`theanets.dataset.Dataset`
            A set of validation data for computing monitor values and
            determining when the loss has stopped improving.

        Returns
        -------
        training : dict
            A dictionary mapping monitor names to values, evaluated on the
            training dataset.
        validation : dict
            A dictionary containing monitor values evaluated on the validation
            dataset.
        '''
        # construct a "shadow" of the input network, using the original
        # network's encoding layers, with tied weights in an autoencoder
        # configuration.
        layers_ = list(self.network.layers[:-1])
        for l in layers_[::-1][:-2]:
            layers_.append(layers.build(
                'tied', partner=l, activation=l.kwargs['activation']))
        layers_.append(layers.build(
            'tied', partner=layers_[1], activation='linear'))

        logging.info('creating shadow network')
        ae = feedforward.Autoencoder(tied_weights=True, layers=layers_)

        # train the autoencoder using the supervised layerwise pretrainer.
        pre = SupervisedPretrainer(ae, *self.args, **self.kwargs)
        for monitors in pre.itertrain(train_set, valid_set=valid_set, **kwargs):
            yield monitors

        logging.info('completed unsupervised pretraining')

# -*- coding: utf-8 -*-

r'''Loss functions for neural network models.
'''

import numpy as np
import theano.sparse as SS
import theano.tensor as TT

from . import util


class Loss(util.Registrar(str('Base'), (), {})):
    r'''A loss function for a neural network model.

    Parameters
    ----------
    in_dim : int
        Number of dimensions required to store the input data to compute the
        loss.
    out_dim : int, optional
        Number of dimensions required to store the target values for computing
        the loss. If this is None (the default), no target values are required
        for this loss.
    weighted : bool, optional
        If True, a floating-point array of weights with the same dimensions as
        ``out_dim`` will be required to compute the "weighted" loss. Defaults
        to False.
    sparse_input : bool or str, optional
        If this is ``'csr'`` or ``'csc'``, then the inputs to the loss will be
        stored as sparse matrices in the CSR or CSC format (respectively). If
        this is True, sparse input will be enabled in CSR format. By default
        this is False, which means inputs are dense.
    output_name : str, optional
        Name of the network output to tap for computing the loss. Defaults to
        'out:out', the name of the default output of the last layer in a linear
        network.

    Raises
    ------
    AssertionError :
        If ``sparse_input`` is enabled and ``in_dim`` is not 2.

    Attributes
    ----------
    input : :class:`theano.TensorVariable`
        A symbolic Theano variable representing input data.
    target : :class:`theano.TensorVariable`
        A symbolic Theano variable representing target output data. None if no
        target values are required to compute the loss.
    weight : :class:`theano.TensorVariable`
        A symbolic Theano variable representing target weights. None if no
        weights are required to compute the loss.
    variables : list of :class:`theano.TensorVariable`
        A list of all variables required to compute the loss.
    output_name : str
        Name of the network output to tap for computing the loss.
    '''

    F_CONTAINERS = (TT.scalar, TT.vector, TT.matrix, TT.tensor3, TT.tensor4)
    I_CONTAINERS = (TT.iscalar, TT.ivector, TT.imatrix, TT.itensor3, TT.itensor4)

    def __init__(self, in_dim, out_dim=None, weighted=False, sparse_input=False,
                 output_name='out'):
        self.input = Loss.F_CONTAINERS[in_dim]('input')
        if sparse_input is True or \
           isinstance(sparse_input, str) and sparse_input.lower() == 'csr':
            assert in_dim == 2, 'Theano only supports sparse arrays with 2 dims'
            self.input = SS.csr_matrix('input')
        if isinstance(sparse_input, str) and sparse_input.lower() == 'csc':
            assert in_dim == 2, 'Theano only supports sparse arrays with 2 dims'
            self.input = SS.csc_matrix('input')
        self.variables = [self.input]
        self.target = None
        if out_dim:
            self.target = Loss.F_CONTAINERS[out_dim]('target')
            self.variables.append(self.target)
        self.weight = None
        if weighted:
            self.weight = Loss.F_CONTAINERS[out_dim or in_dim]('weight')
            self.variables.append(self.weight)
        self.output_name = output_name
        if ':' not in self.output_name:
            self.output_name += ':out'

    def diff(self, outputs):
        '''Compute the symbolic output difference from our target.

        Parameters
        ----------
        outputs : dict of Theano expressions
            A dictionary mapping network output names to Theano expressions
            representing the outputs of a computation graph.

        Returns
        -------
        diff : Theano expression
            The difference between the graph output (as specified by the
            instance ``output_name``) and the target data (if provided) or the
            input data (otherwise).
        '''
        output = outputs[self.output_name]
        target = self.input if self.target is None else self.target
        return output - target

    def __call__(self, outputs):
        '''Construct the computation graph for this loss function.

        Parameters
        ----------
        outputs : dict of Theano expressions
            A dictionary mapping network output names to Theano expressions
            representing the outputs of a computation graph.

        Returns
        -------
        loss : Theano expression
            The values of the loss given the network output.
        '''
        raise NotImplementedError


class MeanSquaredError(Loss):
    r'''Mean-squared-error (MSE) loss function.

    Notes
    -----

    The mean squared error (MSE) loss computes the mean of the squared
    difference between the output of a computation graph
    :math:`x = (x_1, \dots, x_d)` and its expected target value
    :math:`t = (t_1, \dots, t_d)`. Mathematically,

    .. math::
       \begin{eqnarray*}
       \mathcal{L}(x, t) &=& \frac{1}{d} \|x - t\|_2^2 \\
                         &=& \frac{1}{d} \sum_{i=1}^d (x_i - t_i)^2
       \end{eqnarray*}

    Whereas some MSE computations return the sum over dimensions, the MSE here
    is computed as an average over the dimensionality of the data.

    For cases where :math:`x` and :math:`t` are matrices, the MSE computes the
    average over corresponding rows in :math:`x` and :math:`t`.
    '''

    __extra_registration_keys__ = ['MSE']

    def __call__(self, outputs):
        '''Construct the computation graph for this loss function.

        Parameters
        ----------
        outputs : dict of Theano expressions
            A dictionary mapping network output names to Theano expressions
            representing the outputs of a computation graph.

        Returns
        -------
        loss : Theano expression
            The values of the loss given the network output.
        '''
        err = self.diff(outputs)
        if self.weight is not None:
            return (self.weight * err * err).sum() / self.weight.sum()
        return (err * err).mean()


class MeanAbsoluteError(Loss):
    r'''Mean-absolute-error (MAE) loss function.

    Notes
    -----

    The mean absolute error (MAE) loss computes the mean difference between the
    output of a computation graph :math:`x = (x_1, \dots, x_d)` and its expected
    target value :math:`t = (t_1, \dots, t_d)`. Mathematically,

    .. math::
       \begin{eqnarray*}
       \mathcal{L}(x, t) &=& \frac{1}{d} \|x - t\|_1 \\
                         &=& \frac{1}{d} \sum_{i=1}^d |x_i - t_i|
       \end{eqnarray*}

    Whereas some MAE computations return the sum over dimensions, the MAE here
    is computed as an average over the dimensionality of the data.

    For cases where :math:`x` and :math:`t` are matrices, the MAE computes the
    average over corresponding rows in :math:`x` and :math:`t`.
    '''

    __extra_registration_keys__ = ['MAE']

    def __call__(self, outputs):
        '''Construct the computation graph for this loss function.

        Parameters
        ----------
        outputs : dict of Theano expressions
            A dictionary mapping network output names to Theano expressions
            representing the outputs of a computation graph.

        Returns
        -------
        loss : Theano expression
            The values of the loss given the network output.
        '''
        err = self.diff(outputs)
        if self.weight is not None:
            return abs(self.weight * err).sum() / self.weight.sum()
        return abs(err).mean()


class GaussianLogLikelihood(Loss):
    r'''Gaussian Log Likelihood (GLL) loss function.

    Parameters
    ----------
    mean_name : str
        Name of the network graph output to use for the mean of the Gaussian
        distribution.
    covar_name : str
        Name of the network graph output to use for the diagonal covariance of
        the Gaussian distribution.

    Notes
    -----

    This loss computes the negative log-likelihood of the observed target data
    :math:`y` under a Gaussian distribution, where the neural network computes
    the mean :math:`\mu` and the diagonal of the covariance :math:`\Sigma` as a
    function of its input :math:`x`. The loss is given by:

    .. math::
       \mathcal{L}(x, y) = -\log p(y) = -\log p\left(y|\mu(x),\Sigma(x)\right)

    where

    .. math::
       p(y) = p(y|\mu,\Sigma) = \frac{1}{(2\pi)^{n/2}|\Sigma|^{1/2}}
          \exp\left\{-\frac{1}{2}(y-\mu)^\top\Sigma^{-1}(y-\mu) \right\}

    is the Gaussian density function.

    The log density :math:`\log p(y)` can be parameterized more conveniently
    [Gu08]_ as:

    .. math::
       \log p(y|\nu,\Lambda) = a + \eta^\top y - \frac{1}{2} y^\top \Lambda y

    where :math:`\Lambda = \Sigma^{-1}` is the precision,
    :math:`\eta = \Lambda\mu` is the covariance-skewed mean, and
    :math:`a=-\frac{1}{2}\left(n\log 2\pi-\log|\Lambda|+\eta^\top\Lambda\eta\right)`
    contains all constant terms. (These terms are all computed as a function of
    the input, :math:`x`.)

    This implementation of the Gaussian log-likelihood loss approximates
    :math:`\Sigma` using only its diagonal. This makes the precision easy to
    compute because

    .. math::
       \Sigma^{-1} = \Lambda =
          \mbox{diag}(\frac{1}{\sigma_1}, \dots, \frac{1}{\sigma_n})

    is just the matrix containing the multiplicative inverse of the diagonal
    covariance values. Similarly, the log-determinant of the precision is just
    the sum of the logs of the diagonal terms:

    .. math::
       \log|\Lambda|=\sum_{i=1}^n\log\lambda_i=-\sum_{i=1}^n\log\sigma_i.

    The log-likelihood is computed separately for each input-output pair in a
    batch, and the overall likelihood is the mean of these individual values.

    Weighted targets unfortunately do not work with this loss at the moment.

    References
    ----------
    .. [Gu08] Multivariate Gaussian Distribution.
        https://www.cs.cmu.edu/~epxing/Class/10701-08s/recitation/gaussian.pdf
    '''

    __extra_registration_keys__ = ['GLL']

    def __init__(self, mean_name='mean', covar_name='covar', **kwargs):
        super(GaussianLogLikelihood, self).__init__(**kwargs)
        self.mean_name = mean_name
        if ':' not in self.mean_name:
            self.mean_name += ':out'
        self.covar_name = covar_name
        if ':' not in self.covar_name:
            self.covar_name += ':out'

    def __call__(self, outputs):
        '''Construct the computation graph for this loss function.

        Parameters
        ----------
        outputs : dict of Theano expressions
            A dictionary mapping network output names to Theano expressions
            representing the outputs of a computation graph.

        Returns
        -------
        loss : Theano expression
            The values of the loss given the network output.
        '''
        # this code is going to look weird to people who are used to seeing
        # implementations of the gaussian likelihood function. our mean, covar,
        # and self.target arrays are all of shape (batch-size, dimensionality).
        # each of these arrays codes an independent input/output pair for the
        # loss, but they're stacked together in matrices for computational
        # efficiency.
        #
        # what's worse, the covariance is encoded as a vector of the diagonal
        # elements, again one per input/output pair in the batch.
        #
        # the upshot of this is that many operations written traditionally as
        # three dot products (vector-matrix-vector, e.g., x^T \Lambda x) are
        # here written as three elementwise array products (x * prec * x),
        # followed by a sum across the last dimension. this has the added
        # benefit that it will be way faster than dot products, but it looks
        # strange in the code below.
        mean = outputs[self.mean_name]
        covar = outputs[self.covar_name]
        prec = 1 / TT.switch(TT.eq(covar, 0), 1.0, covar)  # prevent nans!
        eta = mean * prec
        logpi = TT.cast(mean.shape[-1] * np.log(2 * np.pi), 'float32')
        logdet = TT.log(prec.sum(axis=-1))
        const = logpi - logdet + (eta * prec * eta).sum(axis=-1)
        squared = (self.target * prec * self.target).sum(axis=-1)
        nll = 0.5 * (const + squared) - (eta * self.target).sum(axis=-1)
        return nll.mean()


class KullbackLeiblerDivergence(Loss):
    r'''The KL divergence loss is computed over probability distributions.

    Notes
    -----

    The KL divergence loss is intended to optimize models that generate
    probability distributions. If the outputs :math:`x_i` of a model represent a
    normalized probability distribution (over the output variables), and the
    targets :math:`t_i` represent a normalized target distribution (over the
    output variables), then the KL divergence is given by:

    .. math::
       \mathcal{L}(x, t) = \frac{1}{d} \sum_{i=1}^d t_i \log \frac{t_i}{x_i}

    Here the KL divergence is computed as a mean value over the output variables
    in the model.
    '''

    __extra_registration_keys__ = ['KL', 'KLD']

    def __call__(self, outputs):
        '''Construct the computation graph for this loss function.

        Parameters
        ----------
        outputs : dict of Theano expressions
            A dictionary mapping network output names to Theano expressions
            representing the outputs of a computation graph.

        Returns
        -------
        loss : Theano expression
            The values of the loss given the network output.
        '''
        output = outputs[self.output_name]
        eps = 1e-8
        t = TT.clip(self.target, eps, 1 - eps)
        kl = t * TT.log(t / TT.clip(output, eps, 1 - eps))
        if self.weight is not None:
            return abs(self.weight * kl).sum() / self.weight.sum()
        return abs(kl).mean()


class CrossEntropy(Loss):
    r'''Cross-entropy (XE) loss function for classifiers.

    Parameters
    ----------
    in_dim : int
        Number of dimensions required to store the input data to compute the
        loss.
    out_dim : int, optional
        Number of dimensions required to store the target values for computing
        the loss. If this is None (the default), no target values are required
        for this loss.
    weighted : bool, optional
        If True, a floating-point array of weights with the same dimensions as
        ``out_dim`` will be required to compute the "weighted" loss. Defaults
        to False.
    sparse_input : str, optional
        If this is ``'csr'`` or ``'csc'``, then the inputs to the loss will be
        stored as sparse matrices in the CSR or CSC format (respectively). By
        default this is None, which means inputs are dense.
    output_name : str, optional
        Name of the network output to tap for computing the loss. Defaults to
        'out:out', the name of the default output of the last layer in a linear
        network.

    Raises
    ------
    AssertionError :
        If ``sparse_input`` is True and ``in_dim`` is not 2.

    Attributes
    ----------
    input : :class:`theano.TensorVariable`
        A symbolic Theano variable representing input data.
    target : :class:`theano.TensorVariable`
        A symbolic Theano variable representing target output data. None if no
        target values are required to compute the loss.
    weight : :class:`theano.TensorVariable`
        A symbolic Theano variable representing target weights. None if no
        weights are required to compute the loss.
    variables : list of :class:`theano.TensorVariable`
        A list of all variables required to compute the loss.
    output_name : str
        Name of the network output to tap for computing the loss.

    Notes
    -----

    The cross-entropy between a "true" distribution over discrete classes
    :math:`p(t)` and a "model" distribution over predicted classes :math:`q(x)`
    is the number of bits needed to store the model distribution, under the
    expectation of the true distribution. Mathematically, this loss computes:

    .. math::
       \mathcal{L}(x, t) = - \sum_{k=1}^K p(t=k) \log q(x=k)

    The loss value is similar to the KL divergence between :math:`p` and
    :math:`q`, but it is specifically aimed at classification models. When using
    this loss, targets are assumed to be integers in the half-open interval
    :math:`[0, k)`; the loss is computed by first taking the log of the model
    distributin and then summing up only the entries in the resulting array
    corresponding to the true class.
    '''

    __extra_registration_keys__ = ['XE']

    def __init__(self, in_dim, out_dim, weighted=False, sparse_input=False,
                 output_name='out:out'):
        self.input = Loss.F_CONTAINERS[in_dim]('input')
        if sparse_input is True or \
           isinstance(sparse_input, str) and sparse_input.lower() == 'csr':
            assert in_dim == 2, 'Theano only supports sparse arrays with 2 dims'
            self.input = SS.csr_matrix('input')
        if isinstance(sparse_input, str) and sparse_input.lower() == 'csc':
            assert in_dim == 2, 'Theano only supports sparse arrays with 2 dims'
            self.input = SS.csc_matrix('input')
        self.target = Loss.I_CONTAINERS[out_dim]('target')
        self.variables = [self.input, self.target]
        self.weight = None
        if weighted:
            self.weight = Loss.F_CONTAINERS[out_dim]('weight')
            self.variables.append(self.weight)
        self.output_name = output_name

    def __call__(self, outputs):
        '''Construct the computation graph for this loss function.

        Parameters
        ----------
        outputs : dict of Theano expressions
            A dictionary mapping network output names to Theano expressions
            representing the outputs of a computation graph.

        Returns
        -------
        loss : Theano expression
            The values of the loss given the network output.
        '''
        output = outputs[self.output_name]
        k = output.shape[-1]
        n = TT.prod(output.shape) // k
        prob = output.reshape((n, k))[TT.arange(n), self.target.reshape((n, ))]
        nlp = -TT.log(TT.clip(prob, 1e-8, 1))
        if self.weight is not None:
            return (self.weight.reshape((n, )) * nlp).sum() / self.weight.sum()
        return nlp.mean()

    def accuracy(self, outputs):
        '''Build a Theano expression for computing the accuracy of graph output.

        Parameters
        ----------
        outputs : dict of Theano expressions
            A dictionary mapping network output names to Theano expressions
            representing the outputs of a computation graph.

        Returns
        -------
        acc : Theano expression
            A Theano expression representing the accuracy of the output compared
            to the target data.
        '''
        output = outputs[self.output_name]
        predict = TT.argmax(output, axis=-1)
        correct = TT.eq(predict, self.target)
        acc = correct.mean()
        if self.weight is not None:
            acc = (self.weight * correct).sum() / self.weight.sum()
        return acc


class Hinge(CrossEntropy):
    r'''Hinge loss function for classifiers.

    Notes
    -----

    The hinge loss as implemented here computes the maximum difference between
    the prediction :math:`q(x=k)` for a class :math:`k` and the prediction
    :math:`q(x=t)` for the correct class :math:`t`:

    .. math::
       \mathcal{L}(x, t) = \max(0, \max_k q(x=k) - q(x=t))

    This loss is zero whenever the prediction for the correct class is the
    largest over classes, and increases linearly when the prediction for an
    incorrect class is the largest.
    '''

    __extra_registration_keys__ = []

    def __call__(self, outputs):
        '''Construct the computation graph for this loss function.

        Parameters
        ----------
        outputs : dict of Theano expressions
            A dictionary mapping network output names to Theano expressions
            representing the outputs of a computation graph.

        Returns
        -------
        loss : Theano expression
            The values of the loss given the network output.
        '''
        output = outputs[self.output_name]
        k = output.shape[-1]
        n = TT.prod(output.shape) // k
        output = output.reshape((n, k))
        true = output[TT.arange(n), self.target.reshape((n, ))]
        err = TT.maximum(0, (output - true[:, None]).max(axis=-1))
        if self.weight is not None:
            return (self.weight.reshape((n, )) * err).sum() / self.weight.sum()
        return err.mean()

# -*- coding: utf-8 -*-

'''These loss functions are available for neural network models.'''

import climate
import numpy as np
import theano.tensor as TT

from . import util

logging = climate.get_logger(__name__)


class Loss(util.Registrar(str('Base'), (), {})):
    r'''A loss function base class.

    Parameters
    ----------
    target : int or Theano variable
        If this is an integer, it specifies the number of dimensions required to
        store the target values for computing the loss. If it is a Theano
        variable, this variable will be used directly to access target values.
    weight : float, optional
        The importance of this loss for the model being trained. Defaults to 1.
    weighted : bool, optional
        If True, a floating-point array of weights with the same dimensions as
        ``target`` will be required to compute the "weighted" loss. Defaults
        to False.
    output_name : str, optional
        Name of the network output to tap for computing the loss. Defaults to
        'out:out', the name of the default output of the last layer in a linear
        network.

    Attributes
    ----------
    weight : float
        The importance of this loss for the model being trained.
    output_name : str
        Name of the network output to tap for computing the loss.
    '''

    def __init__(self, target, weight=1., weighted=False, output_name='out'):
        self.weight = weight

        self._target = (util.FLOAT_CONTAINERS[target]('target')
                        if isinstance(target, int) else target)

        self._weights = None
        if weighted:
            self._weights = util.FLOAT_CONTAINERS[self._target.ndim]('weights')

        self.output_name = output_name
        if ':' not in self.output_name:
            self.output_name += ':out'

    @property
    def variables(self):
        '''A list of Theano variables used in this loss.'''
        result = [self._target]
        if self._weights is not None:
            result.append(self._weights)
        return result

    def log(self):
        '''Log some diagnostic info about this loss.'''
        logging.info('using loss: %s * %s (output %s)',
                     self.weight, self.__class__.__name__, self.output_name)

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
    average over corresponding rows in :math:`x` and :math:`t`:

    .. math::
       \mathcal{L}(X, T) = \frac{1}{dm} \sum_{j=1}^m \sum_{i=1}^d (x_{ji} - t_{ji})^2
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
        err = outputs[self.output_name] - self._target
        if self._weights is not None:
            return (self._weights * err * err).sum() / self._weights.sum()
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
        err = outputs[self.output_name] - self._target
        if self._weights is not None:
            return abs(self._weights * err).sum() / self._weights.sum()
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

    def __init__(self, mean_name='mean', covar_name='covar', covar_eps=1e-3, **kwargs):
        self.mean_name = mean_name
        if ':' not in self.mean_name:
            self.mean_name += ':out'
        self.covar_name = covar_name
        if ':' not in self.covar_name:
            self.covar_name += ':out'
        self.covar_eps = covar_eps
        super(GaussianLogLikelihood, self).__init__(**kwargs)

    def log(self):
        '''Log some diagnostic info about this loss.'''
        logging.info('using loss: %s * %s (mean %s, covar %s)',
                     self.weight, self.__class__.__name__,
                     self.mean_name, self.covar_name)

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
        # and self._target arrays are all of shape (batch-size, dims). each of
        # these arrays codes an independent input/output pair for the loss, but
        # they're stacked together in matrices for computational efficiency.
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
        prec = 1 / (abs(covar) + self.covar_eps)  # prevent nans!
        eta = mean * prec
        logpi = TT.cast(mean.shape[-1] * np.log(2 * np.pi), 'float32')
        logdet = TT.log(prec.sum(axis=-1))
        const = logpi - logdet + (eta * prec * eta).sum(axis=-1)
        squared = (self._target * prec * self._target).sum(axis=-1)
        nll = 0.5 * (const + squared) - (eta * self._target).sum(axis=-1)
        return nll.mean()


class MaximumMeanDiscrepancy(Loss):
    r'''Maximum Mean Discrepancy (MMD) loss function.

    Parameters
    ----------
    kernel : callable or numeric, optional
        A kernel function to call for computing pairwise kernel values. If this
        is a callable, it should take two Theano arrays as arguments and return
        a Theano array. If it is a numeric value, the kernel will be a Gaussian
        with the given value as the bandwidth parameter. Defaults to 1.

    Notes
    -----

    This loss computes the differential between a predicted distribution
    (generated by a network) and an observed distribution (of data within a
    mini-batch). The loss is given by:

    .. math::
       \mathcal{L}(x, y) = \| \sum_{j=1}^N \phi(y_j) - \sum_{i=1}^N \phi(x_i) \|_2^2

    This can be expanded to

    .. math::
       \mathcal{L}(x, y) = \sum_{j=1}^N \sum_{j'=1}^N \phi(y_j)^\top \phi(y_{j'})
          - 2 \sum_{j=1}^N \sum_{i=1}^N \phi(y_j)^\top \phi(x_i)
          + \sum_{i=1}^N \sum_{i'=1}^N \phi(x_i)^\top \phi(x_{i'})

    and then the kernel trick can be applied,

    .. math::
       \mathcal{L}(x, y) = \sum_{j=1}^N \sum_{j'=1}^N k(y_j, y_{j'})
          - 2 \sum_{j=1}^N \sum_{i=1}^N k(y_j, x_i)
          + \sum_{i=1}^N \sum_{i'=1}^N k(x_i, x_{i'})

    By default the loss here uses the Gaussian kernel

    .. math::
       k(x, x') = \exp(-(x-x')^2/\sigma)

    where :math:`\sigma` is a scalar bandwidth parameter. However, other kernels
    can be provided when constructing the loss.

    References
    ----------

    .. [Gre07] A. Gretton, K. M. Borgwardt, M. Rasch, B. Scholkopf, & A. J.
       Smola (NIPS 2007) "A Kernel Method for the Two-Sample-Problem."
       http://papers.nips.cc/paper/3110-a-kernel-method-for-the-two-sample-problem.pdf

    .. [Li15] Y. Li, K. Swersky, & R. Zemel (ICML 2015) "Generative Moment
       Matching Networks." http://jmlr.org/proceedings/papers/v37/li15.pdf
    '''

    __extra_registration_keys__ = ['MMD']

    @staticmethod
    def gaussian(bw):
        def kernel(x, y):
            # this dimshuffle lets us compute squared euclidean distance with a
            # broadcasted subtraction, a square, and a sum.
            r = x.dimshuffle(0, 'x', *tuple(range(1, x.ndim)))
            return TT.exp(TT.sqr(r - y).sum(axis=-1) / -bw)
        return kernel

    def __init__(self, kernel=1, **kwargs):
        super(MaximumMeanDiscrepancy, self).__init__(**kwargs)
        if isinstance(kernel, (int, float)):
            kernel = MaximumMeanDiscrepancy.gaussian(kernel)
        self.kernel = kernel

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
        xx = self.kernel(self._target, self._target)
        xy = self.kernel(self._target, output)
        yy = self.kernel(output, output)
        return xx.mean() - 2 * xy.mean() + yy.mean()


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
        t = TT.clip(self._target, eps, 1 - eps)
        kl = t * TT.log(t / TT.clip(output, eps, 1 - eps))
        if self._weights is not None:
            return abs(self._weights * kl).sum() / self._weights.sum()
        return abs(kl).mean()


class CrossEntropy(Loss):
    r'''Cross-entropy (XE) loss function for classifiers.

    Parameters
    ----------
    target : int
        Number of dimensions required to store the target values for computing
        the loss.
    weight : float, optional
        The importance of this loss for the model being trained. Defaults to 1.
    weighted : bool, optional
        If True, a floating-point array of weights with the same dimensions as
        ``out_dim`` will be required to compute the "weighted" loss. Defaults
        to False.
    output_name : str, optional
        Name of the network output to tap for computing the loss. Defaults to
        'out:out', the name of the default output of the last layer in a linear
        network.

    Attributes
    ----------
    weight : float, optional
        The importance of this loss for the model being trained.
    output_name : str
        Name of the network output to tap for computing the loss.

    Notes
    -----

    The cross-entropy between a "true" distribution over discrete classes
    :math:`p(t)` and a "model" distribution over predicted classes :math:`q(x)`
    is the expected number of bits needed to store the model distribution, under
    the expectation of the true distribution. Mathematically, this loss
    computes:

    .. math::
       \mathcal{L}(x, t) = - \sum_{k=1}^K p(t=k) \log q(x=k)

    The loss value is similar to the KL divergence between :math:`p` and
    :math:`q`, but it is specifically aimed at classification models. When using
    this loss, targets are assumed to be integers in the half-open interval
    :math:`[0, k)`; internally, the loss is computed by first taking the log of
    the model distribution and then summing up only the entries in the resulting
    array corresponding to the true class.
    '''

    __extra_registration_keys__ = ['XE']

    def __init__(self, target, weight=1., weighted=False, output_name='out'):
        super(CrossEntropy, self).__init__(
            target, weight=weight, weighted=weighted, output_name=output_name)
        self._target = util.INT_CONTAINERS[target]('target')

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
        prob = output.reshape((n, k))[TT.arange(n), self._target.reshape((n, ))]
        nlp = -TT.log(TT.clip(prob, 1e-8, 1))
        if self._weights is not None:
            return (self._weights.reshape((n, )) * nlp).sum() / self._weights.sum()
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
        correct = TT.eq(predict, self._target)
        acc = correct.mean()
        if self._weights is not None:
            acc = (self._weights * correct).sum() / self._weights.sum()
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
        true = output[TT.arange(n), self._target.reshape((n, ))]
        err = TT.maximum(0, (output - true[:, None]).max(axis=-1))
        if self._weights is not None:
            return (self._weights.reshape((n, )) * err).sum() / self._weights.sum()
        return err.mean()

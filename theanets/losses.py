'''Loss functions for neural network models.
'''

import theano.sparse as SS
import theano.tensor as TT

from . import util


def build(name, **kwargs):
    '''Construct an activation function by name.

    Parameters
    ----------
    name : str or :class:`Loss`
        The name of the type of loss function to build.
    kwargs : dict
        Additional named arguments to pass to the loss constructor.

    Returns
    -------
    loss : :class:`Loss`
        A neural network loss function instance.
    '''
    return Loss.build(name, **kwargs)


class Loss(util.Registrar(str('Base'), (), {})):
    '''A loss function for a neural network model.

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
    '''

    F_CONTAINERS = (TT.scalar, TT.vector, TT.matrix, TT.tensor3, TT.tensor4)
    I_CONTAINERS = (TT.iscalar, TT.ivector, TT.imatrix, TT.itensor3, TT.itensor4)

    def __init__(self, in_dim, out_dim=None, weighted=False, sparse_input=False):
        self.input = Loss.F_CONTAINERS[in_dim]('input')
        if (sparse_input is True or
            isinstance(sparse_input, str) and sparse_input.lower() == 'csr'):
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

    def diff(self, output):
        '''Compute the symbolic output difference from our target.

        Parameters
        ----------
        output : Theano expression
            A Theano expression representing the output of a computation graph.

        Returns
        -------
        diff : Theano expression
            The difference between the graph output and the target data (if
            provided) or the input data (otherwise).
        '''
        return output - (self.input if self.target is None else self.target)

    def __call__(self, output):
        '''Construct the computation graph for this loss function.

        Parameters
        ----------
        output : Theano expression
            A Theano expression representing the output of a computation graph.

        Returns
        -------
        loss : Theano expression
            The values of the loss given the network output.
        '''
        raise NotImplementedError


class MeanSquaredError(Loss):
    '''Mean-squared-error (MSE) loss function.

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

    def __call__(self, output):
        '''Construct the computation graph for this loss function.

        Parameters
        ----------
        output : Theano expression
            A Theano expression representing the output of a computation graph.

        Returns
        -------
        loss : Theano expression
            The values of the loss given the network output.
        '''
        err = self.diff(output)
        if self.weight is not None:
            return (self.weight * err * err).sum() / self.weight.sum()
        return (err * err).mean()


class MeanAbsoluteError(Loss):
    '''Mean-absolute-error (MAE) loss function.

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

    def __call__(self, output):
        '''Construct the computation graph for this loss function.

        Parameters
        ----------
        output : Theano expression
            A Theano expression representing the output of a computation graph.

        Returns
        -------
        loss : Theano expression
            The values of the loss given the network output.
        '''
        err = self.diff(output)
        if self.weight is not None:
            return abs(self.weight * err).sum() / self.weight.sum()
        return abs(err).mean()


class Hinge(Loss):
    '''Hinge loss function.

    .. math::
       \mathcal{L}(x, t) = \begin{cases}
         x - t \mbox{ if } x > t \\ 0 \mbox{ otherwise} \end{cases}
    '''

    def __call__(self, output):
        '''Construct the computation graph for this loss function.

        Parameters
        ----------
        output : Theano expression
            A Theano expression representing the output of a computation graph.

        Returns
        -------
        loss : Theano expression
            The values of the loss given the network output.
        '''
        err = TT.maximum(0, self.diff(output))
        if self.weight is not None:
            return (self.weight * err).sum() / self.weight.sum()
        return err.mean()


class CrossEntropy(Loss):
    '''Cross-entropy (XE) loss function.

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

    Notes
    -----

    The cross-entropy between a "true" distribution over discrete classes
    :math:`p(t)` and a "model" distribution over classes :math:`q(y)` is the
    number of bits needed to store the model distribution, under the expectation
    of the true distribution. Mathematically, this loss computes:

    .. math::
       \mathcal{L}(x, t) = - \sum_{k=1}^K p(t=k) \log q(x=k)

    The loss value is similar to the KL divergence between :math:`p` and
    :math:`q`.

    When using this loss, targets are assumed to be integers in the half-open
    interval :math:`[0, k)`; the loss is computed by first taking the log of the
    model distributin and then summing up only the entries in the resulting
    array corresponding to the true class.
    '''

    __extra_registration_keys__ = ['XE']

    def __init__(self, in_dim, out_dim, weighted=False, sparse_input=False):
        self.input = Loss.F_CONTAINERS[in_dim]('input')
        if (sparse_input is True or
            isinstance(sparse_input, str) and sparse_input.lower() == 'csr'):
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

    def __call__(self, output):
        '''Construct the computation graph for this loss function.

        Parameters
        ----------
        output : Theano expression
            A Theano expression representing the output of a computation graph.

        Returns
        -------
        loss : Theano expression
            The values of the loss given the network output.
        '''
        k = output.shape[-1]
        n = TT.prod(output.shape)
        prob = output.reshape((-1, k))[
            TT.arange(n // k), self.target.reshape((-1, ))]
        nlp = -TT.log(TT.clip(prob, 1e-8, 1))
        if self.weight is not None:
            return (self.weight.reshape((-1, )) * nlp).sum() / self.weight.sum()
        return nlp.mean()

    def accuracy(self, output):
        '''Build a Theano expression for computing the accuracy of graph output.

        Parameters
        ----------
        output : Theano expression
            An expression representing the output of a computation graph.

        Returns
        -------
        acc : Theano expression
            A Theano expression representing the accuracy of the output compared
            to the target data.
        '''
        predict = TT.argmax(output, axis=-1)
        correct = TT.eq(predict, self.target)
        acc = correct.mean()
        if self.weight is not None:
            acc = (self.weight * correct).sum() / self.weight.sum()
        return acc

# -*- coding: utf-8 -*-

r'''
'''

import numpy as np
import theano
import theano.sparse as SS
import theano.tensor as TT
import warnings

from . import graph
from . import layers


class Autoencoder(graph.Network):
    r'''An autoencoder attempts to reproduce its input.

    Some types of neural network models have been shown to learn useful features
    from a set of data without requiring any label information. This learning
    task is often referred to as feature learning or manifold learning. A class
    of neural network architectures known as autoencoders are ideally suited for
    this task. An autoencoder takes as input a data sample and attempts to
    produce the same data sample as its output. Formally, an autoencoder defines
    a mapping from a source space to itself:

    .. math::
       F_\theta: \mathcal{S} \to \mathcal{S}

    Often, this mapping can be decomposed into an "encoding" stage
    :math:`f_\alpha(\cdot)` and a corresponding "decoding" stage
    :math:`g_\beta(\cdot)` to and from some latent space :math:`\mathcal{Z} =
    \mathbb{R}^{n_z}`:

    .. math::
       \begin{eqnarray*}
       f_\alpha &:& \mathcal{S} \to \mathcal{Z} \\
       g_\beta &:& \mathcal{Z} \to \mathcal{S}
       \end{eqnarray*}

    Autoencoders form an interesting class of models for several reasons. They:

    - require only "unlabeled" data (which is typically easy to obtain),
    - are generalizations of many popular density estimation techniques, and
    - can be used to model the "manifold" or density of a dataset.

    If we have a labeled dataset containing :math:`m` :math:`d`-dimensional
    input samples :math:`X \in \mathbb{R}^{m \times d}`, then the loss that the
    autoencoder model optimizes with respect to the model parameters
    :math:`\theta` is:

    .. math::
       \begin{eqnarray*}
       \mathcal{L}(X, \theta)
         &=&
           \frac{1}{m} \sum_{i=1}^m \| F_\theta(x_i) - x_i \|_2^2
           + R(X, \theta) \\
         &=&
           \frac{1}{m} \sum_{i=1}^m \| g_\beta(f_\alpha(x_i)) - x_i \|_2^2
           + R(X, \alpha, \beta)
       \end{eqnarray*}

    where :math:`R` is a regularization function.

    A generic autoencoder can be defined in ``theanets`` by using the
    :class:`Autoencoder <theanets.feedforward.Autoencoder>` class::

      exp = theanets.Experiment(theanets.Autoencoder)

    The ``layers`` parameter is required to define such a model; it can be
    provided on the command-line by using ``--layers A B C ... A``, or in your
    code::

      exp = theanets.Experiment(
          theanets.Autoencoder,
          layers=(A, B, C, ..., A))

    Autoencoders retain all attributes of the parent :class:`Network
    <graph.Network>` class, but additionally can have "tied weights", if the
    layer configuration is palindromic.
    '''

    def _setup_vars(self, sparse_input):
        '''Setup Theano variables for our network.

        Parameters
        ----------
        sparse_input : bool
            Unused -- theanets does not support autoencoders with sparse input.

        Returns
        -------
        vars : list of theano variables
            A list of the variables that this network requires as inputs.
        '''
        assert not sparse_input, 'Theanets does not support sparse autoencoders!'

        # x represents our network's input (and target outputs).
        self.x = TT.matrix('x')

        # the weight array is provided to ensure that different target values
        # are taken into account with different weights during optimization.
        self.weights = TT.matrix('weights')

        if self.weighted:
            return [self.x, self.weights]
        return [self.x]

    @property
    def tied_weights(self):
        '''A boolean indicating whether this network uses tied weights.'''
        return any('tied' in l.__class__.__name__.lower() for l in self.layers)

    def encode(self, x, layer=None, sample=False):
        '''Encode a dataset using the hidden layer activations of our network.

        Parameters
        ----------
        x : ndarray
            A dataset to encode. Rows of this dataset capture individual data
            points, while columns represent the variables in each data point.

        layer : str, optional
            The name of the hidden layer output to use. By default, we use
            the "middle" hidden layer---for example, for a 4,2,4 or 4,3,2,3,4
            autoencoder, we use the "2" layer (typically named "hid1" or "hid2",
            respectively).

        sample : bool, optional
            If True, then draw a sample using the hidden activations as
            independent Bernoulli probabilities for the encoded data. This
            assumes the hidden layer has a logistic sigmoid activation function.

        Returns
        -------
        ndarray :
            The given dataset, encoded by the appropriate hidden layer
            activation.
        '''
        enc = self.feed_forward(x)[self._find_output(layer)]
        if sample:
            return np.random.binomial(n=1, p=enc).astype(np.uint8)
        return enc

    def decode(self, z, layer=None):
        '''Decode an encoded dataset by computing the output layer activation.

        Parameters
        ----------
        z : ndarray
            A matrix containing encoded data from this autoencoder.
        layer : int or str or :class:`Layer <layers.Layer>`, optional
            The index or name of the hidden layer that was used to encode `z`.

        Returns
        -------
        decoded : ndarray
            The decoded dataset.
        '''
        key = self._find_output(layer)
        if key not in self._functions:
            outputs, updates = self.build_graph()
            self._functions[key] = theano.function(
                [outputs[key]], [outputs[self.output_name()]], updates=updates)
        return self._functions[key](z)[0]

    def _find_output(self, layer):
        '''Find a layer output name for the given layer specifier.

        Parameters
        ----------
        layer : None, int, str, or :class:`theanets.layers.Layer`
            A layer specification. If this is None, the "middle" layer in the
            network will be used (i.e., the layer at the middle index in the
            list of network layers). If this is an integer, the corresponding
            layer in the network's layer list will be used. If this is a string,
            the layer with the corresponding name will be returned.

        Returns
        -------
        name : str
            The fully-scoped output name for the desired layer.
        '''
        if layer is None:
            layer = len(self.layers) // 2
        if isinstance(layer, int):
            layer = self.layers[layer]
        if isinstance(layer, str):
            try:
                layer = [l for l in self.layers if l.name == layer][0]
            except IndexError:
                pass
        if isinstance(layer, layers.Layer):
            layer = layer.output_name()
        return layer

    def score(self, x, w=None):
        '''Compute R^2 coefficient of determination for a given input.

        Parameters
        ----------
        x : ndarray (num-examples, num-inputs)
            An array containing data to be fed into the network. Multiple
            examples are arranged as rows in this array, with columns containing
            the variables for each example.

        Returns
        -------
        r2 : float
            The R^2 correlation between the prediction of this netork and its
            input. This can serve as one measure of the information loss of the
            autoencoder.
        '''
        return super(Autoencoder, self).score(x, x, w=w)


class Regressor(graph.Network):
    r'''A regression model attempts to produce a target output.

    Regression models are trained by optimizing a (possibly regularized) loss
    that centers around some measurement of error with respect to the target
    outputs. This regression model implementation uses the mean squared error.

    If we have a labeled dataset containing :math:`m` :math:`d`-dimensional
    input samples :math:`X \in \mathbb{R}^{m \times d}` and :math:`m`
    :math:`e`-dimensional paired target outputs :math:`Y \in \mathbb{R}^{m
    \times e}`, then the loss that the Regressor model optimizes with respect to
    the model parameters :math:`\theta` is:

    .. math::
       \mathcal{L}(X, Y, \theta) =
         \frac{1}{m} \sum_{i=1}^m \| F_\theta(x_i) - y_i \|_2^2 + R(X, \theta)

    where :math:`F_\theta` is the feedforward function that computes the network
    output, and :math:`R` is a regularization function.

    A regression model requires the following inputs at training time:

    - ``x``: A two-dimensional array of input data. Each row of ``x`` is
      expected to be one data item. Each column of ``x`` holds the measurements
      of a particular input variable across all data items.
    - ``targets``: A two-dimensional array of target output data. Each row of
      ``targets`` is expected to be the target values for a single data item.
      Each column of ``targets`` holds the measurements of a particular output
      variable across all data items.

    The number of rows in ``x`` must be equal to the number of rows of
    ``targets``, but the number of columns in these two arrays may be whatever
    is required for the inputs and outputs of the problem.
    '''

    def _setup_vars(self, sparse_input):
        '''Setup Theano variables for our network.

        Parameters
        ----------
        sparse_input : bool
            If True, create an input variable that can hold a sparse matrix.
            Defaults to False, which assumes all arrays are dense.

        Returns
        -------
        vars : list of theano variables
            A list of the variables that this network requires as inputs.
        '''
        # x represents our network's input.
        self.x = TT.matrix('x')
        if sparse_input:
            self.x = SS.csr_matrix('x')

        # this variable holds the target outputs for input x.
        self.targets = TT.matrix('targets')

        # the weight array is provided to ensure that different target values
        # are taken into account with different weights during optimization.
        self.weights = TT.matrix('weights')

        if self.weighted:
            return [self.x, self.targets, self.weights]
        return [self.x, self.targets]

    def error(self, outputs):
        '''Build a theano expression for computing the network error.

        Parameters
        ----------
        outputs : dict mapping str to theano expression
            A dictionary of all outputs generated by the layers in this network.

        Returns
        -------
        error : theano expression
            A theano expression representing the network error.
        '''
        err = outputs[self.output_name()] - self.targets
        if self.weighted:
            return (self.weights * err * err).sum() / self.weights.sum()
        return (err * err).mean()


class Classifier(graph.Network):
    r'''A classifier attempts to match a 1-hot target output.

    Classification models in ``theanets`` are trained by optimizing a (possibly
    regularized) loss that centers around the categorical cross-entropy. This
    error computes the difference between the distribution generated by the
    classification model and the empirical distribution of the labeled data.

    If we have a labeled dataset containing :math:`m` :math:`d`-dimensional
    input samples :math:`X \in \mathbb{R}^{m \times d}` and :math:`m` paired
    target outputs :math:`Y \in \{0,1,\dots,K-1\}^m`, then the loss that the
    ``Classifier`` model optimizes with respect to the model parameters
    :math:`\theta` is:

    .. math::
       \mathcal{L}(X, Y, \theta) = R(X, \theta) - \frac{1}{m} \sum_{i=1}^m
          \sum_{k=0}^{K-1} p(k | y_i) \log q_\theta(k | x_i)

    Here, :math:`p(k|y_i)` is the probability that example :math:`i` is labeled
    with class :math:`k`; in ``theanets`` classification models, this is 1 if
    :math:`k = y_i` and 0 otherwise---so, in practice, the sum over classes
    reduces to a single term. Next, :math:`q_\theta(k|x_i)` is the probability
    that the model assigns to class :math:`k` given input :math:`x_i`; this
    corresponds to the relevant softmax output from the model. Finally,
    :math:`R` is a regularization function.

    A classifier model requires the following inputs at training time:

    - ``x``: A two-dimensional array of input data. Each row of ``x`` is
      expected to be one data item. Each column of ``x`` holds the measurements
      of a particular input variable across all data items.
    - ``labels``: A one-dimensional array of target labels. Each element of
      ``labels`` is expected to be the class index for a single data item.

    The number of rows in ``x`` must match the number of elements in the
    ``labels`` vector. Additionally, the values in ``labels`` are expected to
    range from 0 to one less than the number of classes in the data being
    modeled. For example, for the MNIST digits dataset, which represents digits
    0 through 9, the labels array contains integer class labels 0 through 9.
    '''

    DEFAULT_OUTPUT_ACTIVATION = 'softmax'
    '''Classifiers set the default activation for the output layer.'''

    def _setup_vars(self, sparse_input):
        '''Setup Theano variables for our network.

        Parameters
        ----------
        sparse_input : bool
            If True, create an input variable that can hold a sparse matrix.
            Defaults to False, which assumes all arrays are dense.

        Returns
        -------
        vars : list of theano variables
            A list of the variables that this network requires as inputs.
        '''
        # x represents our network's input.
        self.x = TT.matrix('x')
        if sparse_input:
            self.x = SS.csr_matrix('x')

        # for a classifier, this specifies the correct labels for a given input.
        self.labels = TT.ivector('labels')

        # and the weights are reshaped to be just a vector.
        self.weights = TT.vector('weights')

        if self.weighted:
            return [self.x, self.labels, self.weights]
        return [self.x, self.labels]

    def error(self, outputs):
        '''Build a theano expression for computing the network error.

        Parameters
        ----------
        outputs : dict mapping str to theano expression
            A dictionary of all outputs generated by the layers in this network.

        Returns
        -------
        error : theano expression
            A theano expression representing the network error.
        '''
        prob = outputs[self.output_name()][
            TT.arange(self.labels.shape[0]), self.labels]
        nlp = -TT.log(TT.clip(prob, 1e-8, 1))
        if self.weighted:
            return (self.weights * nlp).sum() / self.weights.sum()
        return nlp.mean()

    def monitors(self, **kwargs):
        '''Return expressions that should be computed to monitor training.

        Returns
        -------
        monitors : list of (name, expression) pairs
            A list of named monitor expressions to compute for this network.
        '''
        monitors = super(Classifier, self).monitors(**kwargs)
        outputs, _ = self.build_graph(**kwargs)
        return monitors + [('acc', self.accuracy(outputs))]

    def accuracy(self, outputs):
        '''Build a theano expression for computing the network accuracy.

        Parameters
        ----------
        outputs : dict mapping str to theano expression
            A dictionary of all outputs generated by the layers in this network.

        Returns
        -------
        acc : theano expression
            A theano expression representing the network accuracy.
        '''
        predict = TT.argmax(outputs[self.output_name()], axis=-1)
        correct = TT.eq(predict, self.labels)
        acc = correct.mean()
        if self.weighted:
            acc = (self.weights * correct).sum() / self.weights.sum()
        return acc

    def predict(self, x):
        '''Compute a greedy classification for the given set of data.

        Parameters
        ----------
        x : ndarray (num-examples, num-variables)
            An array containing examples to classify. Examples are given as the
            rows in this array.

        Returns
        -------
        k : ndarray (num-examples, )
            A vector of class index values, one per row of input data.
        '''
        return self.feed_forward(x)[self.output_name()].argmax(axis=-1)

    def classify(self, x):
        warnings.warn('please use predict() instead of classify()',
                      DeprecationWarning)
        return self.predict(x)

    def predict_proba(self, x):
        '''Compute class posterior probabilities for the given set of data.

        Parameters
        ----------
        x : ndarray (num-examples, num-variables)
            An array containing examples to predict. Examples are given as the
            rows in this array.

        Returns
        -------
        p : ndarray (num-examples, num-classes)
            An array of class posterior probability values, one per row of input
            data.
        '''
        return self.feed_forward(x)[self.output_name()]

    def predict_logit(self, x):
        '''Compute the logit values that underlie the softmax output.

        Parameters
        ----------
        x : ndarray (num-examples, num-variables)
            An array containing examples to classify. Examples are given as the
            rows in this array.

        Returns
        -------
        l : ndarray (num-examples, num-classes)
            An array of posterior class logit values, one row of logit values
            per row of input data.
        '''
        return self.feed_forward(x)[self.layers[-1].output_name('pre')]

    def score(self, x, y, w=None):
        '''Compute the mean accuracy on a set of labeled data.

        Parameters
        ----------
        x : ndarray (num-examples, num-variables)
            An array containing examples to classify. Examples are given as the
            rows in this array.
        y : ndarray (num-examples, )
            A vector of integer class labels, one for each row of input data.
        w : ndarray (num-examples, )
            A vector of weights, one for each row of input data.

        Returns
        -------
        score : float
            The (possibly weighted) mean accuracy of the model on the data.
        '''
        eq = y == self.predict(x)
        if w is not None:
            return (w * eq).sum() / w.sum()
        return eq.mean()

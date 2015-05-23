# -*- coding: utf-8 -*-

r'''
'''

import climate
import numpy as np
import theano
import theano.tensor as TT
import theano.sparse as SS

from . import graph
from . import layers

logging = climate.get_logger(__name__)

FLOAT = theano.config.floatX


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
        enc = self.feed_forward(x)[self._find_layer(layer)]
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
        key = self._find_layer(layer)
        if key not in self._functions:
            outputs, _, updates = self.build_graph()
            self._functions[key] = theano.function(
                [outputs[key]], [outputs[self.output_name]], updates=updates)
        return self._functions[key](z)[0]

    def _find_layer(self, layer):
        '''
        '''
        if layer is None:
            layer = len(self.layers) // 2
        if isinstance(layer, int):
            layer = self.layers[layer]
        if isinstance(layer, layers.Layer):
            layer = layer.output_name
        if '.' not in layer:
            layer = '{}.out'.format(layer)
        return layer


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
    '''

    def setup_vars(self):
        '''Setup Theano variables for our network.

        Returns
        -------
        vars : list of theano variables
            A list of the variables that this network requires as inputs.
        '''
        super(Regressor, self).setup_vars()

        # this variable holds the target outputs for input x.
        self.targets = TT.matrix('targets')

        if self.weighted:
            return [self.x, self.targets, self.weights]
        return [self.x, self.targets]

    def error(self, output):
        '''Build a theano expression for computing the network error.

        Parameters
        ----------
        output : theano expression
            A theano expression representing the output of the network.

        Returns
        -------
        error : theano expression
            A theano expression representing the network error.
        '''
        err = output - self.targets
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
    target outputs :math:`Y \in \mathbb{R}^m`, then the loss that the
    ``Classifier`` model optimizes with respect to the model parameters
    :math:`\theta` is:

    .. math::
       \mathcal{L}(X, Y, \theta) = \frac{1}{m} \sum_{i=1}^m -\log F_\theta(x_i)_{y_i} + R(X, \theta)

    where :math:`F_\theta` is the softmax output generated by the classification
    model and :math:`R` is a regularization function.
    '''

    DEFAULT_OUTPUT_ACTIVATION = 'softmax'

    def setup_vars(self):
        '''Setup Theano variables for our network.

        Returns
        -------
        vars : list of theano variables
            A list of the variables that this network requires as inputs.
        '''
        super(Classifier, self).setup_vars()

        # for a classifier, this specifies the correct labels for a given input.
        self.labels = TT.ivector('labels')

        # and the weights are reshaped to be just a vector.
        self.weights = TT.vector('weights')

        if self.weighted:
            return [self.x, self.labels, self.weights]
        return [self.x, self.labels]

    def error(self, output):
        '''Build a theano expression for computing the network error.

        Parameters
        ----------
        output : theano expression
            A theano expression representing the output of the network.

        Returns
        -------
        error : theano expression
            A theano expression representing the network error.
        '''
        lo = TT.cast(1e-5, FLOAT)
        hi = TT.cast(1, FLOAT)
        prob = output[TT.arange(self.labels.shape[0]), self.labels]
        nlp = -TT.log(TT.clip(prob, lo, hi))
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
        outputs, monitors, _ = self.build_graph(**kwargs)
        out = outputs[self.output_name]
        return [('err', self.error(out)), ('acc', self.accuracy(out))] + monitors

    def accuracy(self, output):
        '''Build a theano expression for computing the network accuracy.

        Parameters
        ----------
        output : theano expression
            A theano expression representing the output of the network.

        Returns
        -------
        acc : theano expression
            A theano expression representing the network accuracy.
        '''
        correct = TT.eq(TT.argmax(output, axis=-1), self.labels)
        acc = correct.mean()
        if self.weighted:
            acc = (self.weights * correct).sum() / self.weights.sum()
        return TT.cast(100, FLOAT) * acc

    def classify(self, x):
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
        return self.feed_forward(x)[self.output_name].argmax(axis=-1)

    predict = classify

    predict_proba = graph.Network.predict

    def predict_logit(self, x):
        '''Compute the logit values that underlie the softmax output.

        Parameters
        ----------
        x : ndarray (num-examples, num-variables)
            An array containing examples to classify. Examples are given as the
            rows in this array.

        Returns
        -------
        k : ndarray (num-examples, num-classes)
            An array of class logit values, one row of logit values per row of
            input data.
        '''
        return self.feed_forward(x)['{}.pre'.format(self.layers[-1].name)]

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
        eq = y == self.classify(x)
        if self.weighted and w is not None:
            return (w * eq).sum() / w.sum()
        return eq.mean()

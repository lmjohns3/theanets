# -*- coding: utf-8 -*-

r'''This module contains a number of classes for modeling neural nets in Theano.

Neural networks are really just a concise, computational way of describing a
mathematical model of some data. Before getting into the models below, we'll
first set up the ideas and notation that are used on this page.

At a high level, a feedforward neural network describes a parametric mapping

.. math::
   F_\theta: \mathcal{S} \to \mathcal{T}

between a source space :math:`\mathcal{S}` and a target space
:math:`\mathcal{T}`, using parameters :math:`\theta`. For the MNIST digits, for
example we could think of :math:`\mathcal{S} = \mathbb{R}^{28 \times 28} =
\mathbb{R}^{784}` (i.e., the space of all 28×28 images), and for classifying the
MNIST digits we could think of :math:`\mathcal{T} = \mathbb{R}^{10}`.

This mapping is assumed to be fairly complex. If it were not -- if you could
capture the mapping using a simple expression like :math:`F_a(x) = ax^2` -- then
we would just use the expression directly and not need to deal with an entire
network. So if the mapping is complex, we will do a couple of things to make our
problem tractable. First, we will assume some structure for :math:`F_\theta`.
Second, we will fit our model to some set of data that we have obtained, so that
our parameters :math:`\theta` are tuned to the problem at hand.

Graph structure
---------------

.. image:: _static/feedforward_layers.svg

The mapping :math:`F_\theta` is implemented in neural networks by assuming a
specific, layered form. Computation nodes -- also called units or (sometimes)
neurons -- are arranged in a :math:`k+1` partite graph, with layer :math:`k`
containing :math:`n_k` nodes. The number of input nodes in the graph is referred
to below as :math:`n_0`.

A **weight matrix** :math:`W^k \in \mathbb{R}^{n_{k-1} \times n_k}` specifies
the strength of the connection between nodes in layer :math:`k` and those in
layer :math:`k-1` -- all other pairs of nodes are typically not connected. Each
layer of nodes also has a **bias vector** that determines the offset of each
node from the origin. Together, the parameters :math:`\theta` of the model are
these :math:`k` weight matrices and :math:`k` bias vectors (there are no weights
or biases for the input nodes in the graph).

Local computation
-----------------

.. image:: _static/feedforward_neuron.svg

In a standard feedforward network, each node :math:`i` in layer :math:`k`
receives inputs from all nodes in layer :math:`k-1`, then transforms the
weighted sum of these inputs:

.. math::
   z_i^k = \sigma\left( b_i^k + \sum_{j=1}^{n_{k-1}} w^k_{ji} z_j^{k-1} \right)

where :math:`\sigma: \mathbb{R} \to \mathbb{R}` is an "activation function."
Although many functions will work, typical choices of the activation function
are:

:linear: :math:`\sigma(z) = z`
:rectified linear: :math:`\sigma(z) = \max(0, z)`
:logistic sigmoid: :math:`\sigma(z) = (1 + e^{-z})^{-1}`.

Most activation functions are chosen to incorporate a nonlinearity, since a
model with even multiple linear layers cannot capture nonlinear phenomena. Nodes
in the input layer are assumed to have linear activation (i.e., the input nodes
simply represent the state of the input data), and nodes in the output layer
might have linear or nonlinear activations depending on the modeling task.

Usually all hidden nodes in a network share the same activation function, but
this is not required.
'''

import climate
import numpy as np
import theano
import theano.tensor as TT

from .graph import Network

logging = climate.get_logger(__name__)

FLOAT = theano.config.floatX


class Autoencoder(Network):
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
       \mathcal{L}(X, \theta) &=& \frac{1}{m} \sum_{i=1}^m \| F_\theta(x_i) - x_i \|_2^2 + R(X, \theta) \\
       &=& \frac{1}{m} \sum_{i=1}^m \| g_\beta(f_\alpha(x_i)) - x_i \|_2^2 + R(X, \alpha, \beta)
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

    Autoencoders retain all attributes of the parent :class:`Network` class,
    but additionally can have "tied weights", if the layer configuration is
    palindromic.

    Attributes
    ----------
    tied_weights : bool, optional
        Construct decoding weights using the transpose of the encoding weights
        on corresponding layers. Defaults to False, which means decoding weights
        will be constructed using a separate weight matrix.
    '''

    def setup_decoder(self):
        '''Set up weights for the decoder layers of an autoencoder.

        This implementation allows for decoding weights to be tied to encoding
        weights. If `tied_weights` is False, the decoder is set up using
        :func:`Network.setup_decoder`; if True, then the decoder is set up to be
        a mirror of the encoding layers, using transposed weights.

        Parameters
        ----------
        tied_weights : bool, optional
            If True, use decoding weights that are "tied" to the encoding
            weights. This only makes sense for a limited set of "autoencoder"
            layer configurations. Defaults to False.

        Returns
        -------
        count : int
            A count of the number of tunable decoder parameters.
        '''
        if not self.tied_weights:
            return super(Autoencoder, self).setup_decoder()
        kw = {}
        kw.update(self.kwargs)
        for i in range(len(self.layers) - 1, 1, -1):
            self.layers.append(layers.build('tied', self.layers[i], **kw))
        kw = {}
        kw.update(self.kwargs)
        kw.update(activation=self.output_activation)
        self.layers.append(layers.build('tied', self.layers[1], **kw))

    @property
    def encoding_layers(self):
        '''Compute the layers that will be part of the network encoder.

        This implementation ensures that --layers is compatible with
        --tied-weights; if so, and if the weights are tied, then the encoding
        layers are the first half of the layers in the network. If not, or if
        the weights are not to be tied, then all but the final layer is
        considered an encoding layer.

        Returns
        -------
        layers : list of int
            A list of integers specifying sizes of the encoder network layers.
        '''
        if not self.tied_weights:
            return super(Autoencoder, self).encoding_layers
        error = 'with --tied-weights, --layers must be an odd-length palindrome'
        sizes = []
        for layer in self.kwargs['layers']:
            if isinstance(layer, layers.Layer):
                sizes.append(layer.nout)
            if isinstance(layer, int):
                sizes.append(layer)
            if isinstance(layer, dict):
                sizes.append(layer.get('size', layer.get('nout', -1)))
        assert len(sizes) % 2 == 1, error
        k = len(sizes) // 2
        encode = np.asarray(sizes[:k])
        decode = np.asarray(sizes[k+1:])
        assert (encode == decode[::-1]).all(), error
        return self.kwargs['layers'][:k+1]

    @property
    def tied_weights(self):
        '''A boolean indicating whether this network uses tied weights.'''
        return self.kwargs.get('tied_weights', False)

    def encode(self, x, layer=None, sample=False):
        '''Encode a dataset using the hidden layer activations of our network.

        Parameters
        ----------
        x : ndarray
            A dataset to encode. Rows of this dataset capture individual data
            points, while columns represent the variables in each data point.

        layer : int, optional
            The index of the hidden layer activation to use. By default, we use
            the "middle" hidden layer---for example, for a 4,2,4 or 4,3,2,3,4
            autoencoder, we use the "2" layer (index 1 or 2, respectively).

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
        enc = self.feed_forward(x)[(layer or len(self.layers) // 2)]
        if sample:
            return np.random.binomial(n=1, p=enc).astype(np.uint8)
        return enc

    def decode(self, z, layer=None):
        '''Decode an encoded dataset by computing the output layer activation.

        Parameters
        ----------
        z : ndarray
            A matrix containing encoded data from this autoencoder.

        layer : int, optional
            The index of the hidden layer that was used to encode `z`.

        Returns
        -------
        ndarray :
            The decoded dataset.
        '''
        if not hasattr(self, '_decoders'):
            self._decoders = {}
        layer = layer or len(self.layers) // 2
        if layer not in self._decoders:
            outputs, _, updates = self.build_graph()
            self._decoders[layer] = theano.function(
                [outputs[layer]], [outputs[-1]], updates=updates)
        return self._decoders[layer](z)[0]


class Regressor(Network):
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
       \mathcal{L}(X, Y, \theta) = \frac{1}{m} \sum_{i=1}^m \| F_\theta(x_i) - y_i \|_2^2 + R(X, \theta)

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

        if self.is_weighted:
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
        if self.is_weighted:
            return (self.weights * err * err).sum() / self.weights.sum()
        return (err * err).mean()


class Classifier(Network):
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

    @property
    def output_activation(self):
        '''A string representing the output activation for this network.'''
        return 'softmax'

    def extra_monitors(self, outputs):
        '''Construct extra monitors for this network.

        Parameters
        ----------
        outputs : list of theano expressions
            A list of theano expressions describing the activations of each
            layer in the network.

        Returns
        -------
        monitors : sequence of (name, expression) tuples
            A sequence of named monitor quantities.
        '''
        yield 'acc', self.accuracy(outputs[-1])

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

        if self.is_weighted:
            return [self.x, self.labels, self.weights]
        return [self.x, self.labels]

    @property
    def output_activation(self):
        return 'softmax'

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
        if self.is_weighted:
            return (self.weights * nlp).sum() / self.weights.sum()
        return nlp.mean()

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
        correct = TT.eq(TT.argmax(output, axis=1), self.labels)
        acc = correct.mean()
        if self.is_weighted:
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
        return self.predict(x).argmax(axis=-1)

# -*- coding: utf-8 -*-

'''This module contains common feedforward network models.'''

import numpy as np
import theano
import warnings

from . import graph
from . import layers
from . import regularizers
from . import util


class Autoencoder(graph.Network):
    r'''An autoencoder network attempts to reproduce its input.

    Examples
    --------

    To create an autoencoder, just create a new model instance. Often you'll
    provide the layer configuration at this time:

    >>> model = theanets.Autoencoder([10, 20, 10])

    If you want to create an autoencoder with tied weights, specify that layer
    type when creating the model:

    >>> model = theanets.Autoencoder([10, 20, (10, 'tied')])

    See :ref:`guide-creating` for more information.

    *Data*

    Training data for an autoencoder takes the form of a two-dimensional array.
    The shape of this array is (num-examples, num-variables): the first axis
    enumerates data points in a batch, and the second enumerates the variables
    in the model.

    For instance, to create a training dataset containing 1000 examples:

    >>> inputs = np.random.randn(1000, 10).astype('f')

    *Training*

    Training the model can be as simple as calling the :func:`train()
    <theanets.graph.Network.train>` method:

    >>> model.train([inputs])

    See :ref:`guide-training` for more information about training.

    *Use*

    A model can be used to :func:`predict() <theanets.graph.Network.predict>`
    the output of some input data points:

    >>> test = np.random.randn(3, 10).astype('f')
    >>> print(model.predict(test))

    Additionally, autoencoders can :func:`encode()
    <theanets.feedforward.Autoencoder.encode>` a set of input data points:

    >>> enc = model.encode(test)
    >>> enc.shape
    (3, 20)

    The model can also :func:`decode()
    <theanets.feedforward.Autoencoder.decode>` a set of encoded data:

    >>> model.decode(enc)

    See :ref:`guide-using` for more information about using models.

    Notes
    -----

    Autoencoder models default to a :class:`MSE
    <theanets.losses.MeanSquaredError>` loss. To use a different loss, provide a
    non-default argument for the ``loss`` keyword argument when constructing
    your model.

    Formally, an autoencoder defines a parametric mapping from a data space to
    the same space:

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

    Many extremely common dimensionality reduction techniques can be expressed
    as autoencoders. For instance, Principal Component Analysis (PCA) can be
    expressed as a model with two tied, linear layers:

    >>> pca = theanets.Autoencoder([10, (5, 'linear'), (10, 'tied')])

    Similarly, Independent Component Analysis (ICA) can be expressed as the same
    model, but trained with a sparsity penalty on the hidden-layer activations:

    >>> ica = pca
    >>> ica.train([inputs], hidden_l1=0.1)

    In this light, "nonlinear PCA" is quite easy to formulate as well!
    '''

    def __init__(self, layers, loss='mse', weighted=False, rng=13):
        super(Autoencoder, self).__init__(layers, rng=rng)
        self.set_loss(form=loss, target=self.inputs[0], weighted=weighted)

    def encode(self, x, layer=None, sample=False, **kwargs):
        '''Encode a dataset using the hidden layer activations of our network.

        Parameters
        ----------
        x : ndarray
            A dataset to encode. Rows of this dataset capture individual data
            points, while columns represent the variables in each data point.

        layer : str, optional
            The name of the hidden layer output to use. By default, we use
            the "middle" hidden layer---for example, for a 4,2,4 or 4,3,2,3,4
            autoencoder, we use the layer with size 2.

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
        enc = self.feed_forward(x, **kwargs)[self._find_output(layer)]
        if sample:
            return np.random.binomial(n=1, p=enc).astype(np.uint8)
        return enc

    def decode(self, z, layer=None, **kwargs):
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
            regs = regularizers.from_kwargs(self, **kwargs)
            outputs, updates = self.build_graph(regs)
            self._functions[key] = theano.function(
                [outputs[key]],
                [outputs[self.layers[-1].output_name]],
                updates=updates)
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
        if isinstance(layer, util.basestring):
            try:
                layer = [l for l in self.layers if l.name == layer][0]
            except IndexError:
                pass
        if isinstance(layer, layers.Layer):
            layer = layer.output_name
        return layer

    def score(self, x, w=None, **kwargs):
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
        return super(Autoencoder, self).score(x, x, w=w, **kwargs)


class Regressor(graph.Network):
    '''A regressor attempts to produce a target output given some inputs.

    Examples
    --------

    To create a regression model, just create a new class instance. Often you'll
    provide the layer configuration at this time:

    >>> model = theanets.Regressor([10, 20, 3])

    See :ref:`guide-creating` for more information.

    *Data*

    Training data for a regression model takes the form of two two-dimensional
    arrays. The shapes of both of these arrays are (num-examples, num-variables)
    -- the first axis enumerates data points in a batch, and the second
    enumerates the relevant variables (input variables for the input array, and
    output variables for the output array).

    For instance, to create a training dataset containing 1000 examples:

    >>> inputs = np.random.randn(1000, 10).astype('f')
    >>> outputs = np.random.randn(1000, 3).astype('f')

    *Training*

    Training the model can be as simple as calling the :func:`train()
    <theanets.graph.Network.train>` method, with the inputs and target outputs
    as data:

    >>> model.train([inputs, outputs])

    See :ref:`guide-training` for more information.

    *Use*

    A regression model can be used to :func:`predict()
    <theanets.graph.Network.predict>` the output of some input data points:

    >>> test = np.random.randn(3, 10).astype('f')
    >>> print(model.predict(test))

    See :ref:`guide-using` for more information.

    Notes
    -----

    Regressor models default to a :class:`MSE
    <theanets.losses.MeanSquaredError>` loss. To use a different loss, provide a
    non-default argument for the ``loss`` keyword argument when constructing
    your model.
    '''


class Classifier(graph.Network):
    '''A classifier computes a distribution over labels, given an input.

    Examples
    --------

    To create a classification model, just create a new class instance. Often
    you'll provide the layer configuration at this time:

    >>> model = theanets.Classifier([10, (20, 'tanh'), 50])

    See :ref:`guide-creating` for more information.

    *Data*

    Training data for a classification model takes the form of a two-dimensional
    array of input data and a one-dimensional vector of target labels. The input
    array has a shape (num-examples, num-variables): the first axis enumerates
    data points in a batch, and the second enumerates the input variables in the
    model.

    The second array provides the target class labels for the inputs. Its shape
    is (num-examples, ), and each integer value in the array gives the class
    label for the corresponding input example.

    For instance, to create a training dataset containing 1000 examples:

    >>> inputs = np.random.randn(1000, 10).astype('f')
    >>> outputs = np.random.randint(50, size=1000).astype('i')

    *Training*

    Training the model can be as simple as calling the :func:`train()
    <theanets.graph.Network.train>` method, giving the inputs and target outputs
    as a dataset:

    >>> model.train([inputs, outputs])

    See :ref:`guide-training` for more information.

    *Use*

    A classification model can be used to :func:`predict()
    <theanets.graph.Network.predict>` the output of some input data points:

    >>> test = np.random.randn(3, 10).astype('f')
    >>> print(model.predict(test))

    This method returns a vector containing the most likely class for each input
    example.

    To retrieve the probabilities of the classes for each example, use
    :func:`predict_proba() <theanets.feedforward.Classifier.predict_proba>`:

    >>> model.predict_proba(test).shape
    (3, 50)

    See also :ref:`guide-using` for more information.

    Notes
    -----

    Classifier models default to a :class:`cross-entropy
    <theanets.losses.CrossEntropy>` loss. To use a different loss, provide a
    non-default argument for the ``loss`` keyword argument when constructing
    your model.
    '''

    DEFAULT_OUTPUT_ACTIVATION = 'softmax'
    '''Default activation for the output layer.'''

    OUTPUT_NDIM = 1
    '''Number of dimensions for holding output data arrays.'''

    def __init__(self, layers, loss='xe', weighted=False, rng=13):
        super(Classifier, self).__init__(layers, loss=loss, weighted=weighted, rng=rng)

    def monitors(self, **kwargs):
        '''Return expressions that should be computed to monitor training.

        Returns
        -------
        monitors : list of (name, expression) pairs
            A list of named monitor expressions to compute for this network.
        '''
        monitors = super(Classifier, self).monitors(**kwargs)
        regs = regularizers.from_kwargs(self, **kwargs)
        outputs, _ = self.build_graph(regs)
        return monitors + [('acc', self.losses[0].accuracy(outputs))]

    def predict(self, x, **kwargs):
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
        outputs = self.feed_forward(x, **kwargs)
        return outputs[self.layers[-1].output_name].argmax(axis=-1)

    def classify(self, x, **kwargs):
        warnings.warn('please use predict() instead of classify()',
                      DeprecationWarning)
        return self.predict(x, **kwargs)

    def predict_proba(self, x, **kwargs):
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
        return self.feed_forward(x, **kwargs)[self.layers[-1].output_name]

    def predict_logit(self, x, **kwargs):
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
        return self.feed_forward(x, **kwargs)[self.layers[-1].full_name('pre')]

    def score(self, x, y, w=None, **kwargs):
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
        eq = y == self.predict(x, **kwargs)
        if w is not None:
            return (w * eq).sum() / w.sum()
        return eq.mean()

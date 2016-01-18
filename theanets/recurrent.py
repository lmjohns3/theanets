# -*- coding: utf-8 -*-

'''This module contains recurrent network models and utilities.'''

import collections
import numpy as np
import re

from . import feedforward


def batches(arrays, steps=100, batch_size=64, rng=None):
    '''Create a callable that generates samples from a dataset.

    Parameters
    ----------
    arrays : list of ndarray (time-steps, data-dimensions)
        Arrays of data. Rows in these arrays are assumed to correspond to time
        steps, and columns to variables. Multiple arrays can be given; in such
        a case, these arrays usually correspond to [input, output]---for
        example, for a recurrent regression problem---or [input, output,
        weights]---for a weighted regression or classification problem.
    steps : int, optional
        Generate samples of this many time steps. Defaults to 100.
    batch_size : int, optional
        Generate this many samples per call. Defaults to 64. This must match the
        batch_size parameter that was used when creating the recurrent network
        that will process the data.
    rng : :class:`numpy.random.RandomState` or int, optional
        A random number generator, or an integer seed for a random number
        generator. If not provided, the random number generator will be created
        with an automatically chosen seed.

    Returns
    -------
    callable :
        A callable that can be used inside a dataset for training a recurrent
        network.
    '''
    assert batch_size >= 2, 'batch_size must be at least 2!'
    assert isinstance(arrays, (tuple, list)), 'arrays must be a tuple or list!'

    if rng is None or isinstance(rng, int):
        rng = np.random.RandomState(rng)

    def sample():
        xs = [np.zeros((batch_size, steps, a.shape[1]), a.dtype) for a in arrays]
        for i in range(batch_size):
            j = rng.randint(len(arrays[0]) - steps)
            for x, a in zip(xs, arrays):
                x[i] = a[j:j+steps]
        return xs

    return sample


class Text(object):
    '''A class for handling sequential text data.

    Parameters
    ----------
    text : str
        A blob of text.
    alpha : str, optional
        An alphabet to use for representing characters in the text. If not
        provided, all characters from the text occurring at least ``min_count``
        times will be used.
    min_count : int, optional
        If the alphabet is to be computed from the text, discard characters that
        occur fewer than this number of times. Defaults to 2.
    unknown : str, optional
        A character to use to represent "out-of-alphabet" characters in the
        text. This must not be in the alphabet. Defaults to '\0'.

    Attributes
    ----------
    text : str
        A blob of text, with all non-alphabet characters replaced by the
        "unknown" character.
    alpha : str
        A string containing each character in the alphabet.
    '''

    def __init__(self, text, alpha=None, min_count=2, unknown='\0'):
        self.alpha = alpha
        if self.alpha is None:
            self.alpha = ''.join(sorted(set(
                char for char, count in
                collections.Counter(text).items()
                if char != unknown and count >= min_count)))
        self.text = re.sub(r'[^{}]'.format(re.escape(self.alpha)), unknown, text)
        assert unknown not in self.alpha
        self._rev_index = unknown + self.alpha
        self._fwd_index = dict(zip(self._rev_index, range(1 + len(self.alpha))))

    def encode(self, txt):
        '''Encode a text string by replacing characters with alphabet index.

        Parameters
        ----------
        txt : str
            A string to encode.

        Returns
        -------
        classes : list of int
            A sequence of alphabet index values corresponding to the given text.
        '''
        return list(self._fwd_index.get(c, 0) for c in txt)

    def decode(self, enc):
        '''Encode a text string by replacing characters with alphabet index.

        Parameters
        ----------
        classes : list of int
            A sequence of alphabet index values to convert to text.

        Returns
        -------
        txt : str
            A string containing corresponding characters from the alphabet.
        '''
        return ''.join(self._rev_index[c] for c in enc)

    def classifier_batches(self, steps, batch_size, rng=None):
        '''Create a callable that returns a batch of training data.

        Parameters
        ----------
        steps : int
            Number of time steps in each batch.
        batch_size : int
            Number of training examples per batch.
        rng : :class:`numpy.random.RandomState` or int, optional
            A random number generator, or an integer seed for a random number
            generator. If not provided, the random number generator will be
            created with an automatically chosen seed.

        Returns
        -------
        batch : callable
            A callable that, when called, returns a batch of data that can be
            used to train a classifier model.
        '''
        assert batch_size >= 2, 'batch_size must be at least 2!'

        if rng is None or isinstance(rng, int):
            rng = np.random.RandomState(rng)

        T = np.arange(steps)

        def batch():
            inputs = np.zeros((batch_size, steps, 1 + len(self.alpha)), 'f')
            outputs = np.zeros((batch_size, steps), 'i')
            for b in range(batch_size):
                offset = rng.randint(len(self.text) - steps - 1)
                enc = self.encode(self.text[offset:offset + steps + 1])
                inputs[b, T, enc[:-1]] = 1
                outputs[b, T] = enc[1:]
            return [inputs, outputs]

        return batch


class Autoencoder(feedforward.Autoencoder):
    '''An autoencoder network attempts to reproduce its input.

    Examples
    --------

    To create a recurrent autoencoder, just create a new model instance. Often
    you'll provide the layer configuration at this time:

    >>> model = theanets.recurrent.Autoencoder([10, (20, 'rnn'), 10])

    See :ref:`guide-creating` for more information.

    *Data*

    Training data for a recurrent autoencoder takes the form of a
    three-dimensional array. The shape of this array is (num-examples,
    num-time-steps, num-variables): the first axis enumerates data points in a
    batch, the second enumerates time steps, and the third enumerates the
    variables in the model.

    For instance, to create a training dataset containing 1000 examples, each
    with 100 time steps:

    >>> inputs = np.random.randn(1000, 100, 10).astype('f')

    *Training*

    Training the model can be as simple as calling the :func:`train()
    <theanets.graph.Network.train>` method:

    >>> model.train([inputs])

    See :ref:`guide-training` for more information.

    *Use*

    A model can be used to :func:`predict() <theanets.graph.Network.predict>`
    the output of some input data points:

    >>> test = np.random.randn(3, 200, 10).astype('f')
    >>> print(model.predict(test))

    Note that the test data does not need to have the same number of time steps
    as the training data.

    Additionally, autoencoders can :func:`encode()
    <theanets.feedforward.Autoencoder.encode>` a set of input data points:

    >>> enc = model.encode(test)

    See :ref:`guide-using` for more information.

    Notes
    -----

    Autoencoder models default to a :class:`MSE
    <theanets.losses.MeanSquaredError>` loss. To use a different loss, provide a
    non-default argument for the ``loss`` keyword argument when constructing
    your model.
    '''

    INPUT_NDIM = 3
    '''Number of dimensions for holding input data arrays.'''

    OUTPUT_NDIM = 3
    '''Number of dimensions for holding output data arrays.'''


class Regressor(feedforward.Regressor):
    '''A regressor attempts to produce a target output given some inputs.

    Examples
    --------

    To create a recurrent regression model, just create a new class instance.
    Often you'll provide the layer configuration at this time:

    >>> model = theanets.recurrent.Regressor([10, (20, 'rnn'), 3])

    See :ref:`guide-creating` for more information.

    *Data*

    Training data for a recurrent regression model takes the form of two
    three-dimensional arrays. The shapes of these arrays are (num-examples,
    num-time-steps, num-variables): the first axis enumerates data points in a
    batch, the second enumerates time steps, and the third enumerates the
    variables (input variables for the input array, and output variables for the
    output array) in the model.

    For instance, to create a training dataset containing 1000 examples, each
    with 100 time steps:

    >>> inputs = np.random.randn(1000, 100, 10).astype('f')
    >>> outputs = np.random.randn(1000, 100, 3).astype('f')

    *Training*

    Training the model can be as simple as calling the :func:`train()
    <theanets.graph.Network.train>` method:

    >>> model.train([inputs, outputs])

    See :ref:`guide-training` for more information.

    *Use*

    A model can be used to :func:`predict() <theanets.graph.Network.predict>`
    the output of some input data points:

    >>> test = np.random.randn(3, 200, 10).astype('f')
    >>> print(model.predict(test))

    Note that the test data does not need to have the same number of time steps
    as the training data.

    See :ref:`guide-using` for more information.

    Notes
    -----

    Regressor models default to a :class:`MSE
    <theanets.losses.MeanSquaredError>` loss. To use a different loss, provide a
    non-default argument for the ``loss`` keyword argument when constructing
    your model.
    '''

    INPUT_NDIM = 3
    '''Number of dimensions for holding input data arrays.'''

    OUTPUT_NDIM = 3
    '''Number of dimensions for holding output data arrays.'''


class Classifier(feedforward.Classifier):
    '''A classifier computes a distribution over labels, given an input.

    Examples
    --------

    To create a recurrent classification model, just create a new class
    instance. Often you'll provide the layer configuration at this time:

    >>> model = theanets.recurrent.Classifier([10, (20, 'rnn'), 50])

    See :ref:`guide-creating` for more information.

    *Data*

    Training data for a recurrent classification model takes the form of two
    three-dimensional arrays.

    The first array provides the input data for the model. Its shape is
    (num-examples, num-time-steps, num-variables): the first axis enumerates
    data points in a batch, the second enumerates time steps, and the third
    enumerates the input variables in the model.

    The second array provides the target class labels for the inputs. Its shape
    is (num-examples, num-time-steps), and each integer value in the array gives
    the class label for the corresponding input example and time step.

    For instance, to create a training dataset containing 1000 examples, each
    with 100 time steps:

    >>> inputs = np.random.randn(1000, 100, 10).astype('f')
    >>> outputs = np.random.randint(50, size=(1000, 100)).astype('i')

    *Training*

    Training the model can be as simple as calling the :func:`train()
    <theanets.graph.Network.train>` method:

    >>> model.train([inputs, outputs])

    See :ref:`guide-training` for more information.

    *Use*

    A model can be used to :func:`predict() <theanets.graph.Network.predict>`
    the output of some input data points:

    >>> test = np.random.randn(3, 200, 10).astype('f')
    >>> print(model.predict(test))

    This method returns a two-dimensional array containing the most likely class
    for each input example and time step.

    Note that the test data does not need to have the same number of time steps
    as the training data.

    To retrieve the probabilities of the classes for each example, use
    :func:`predict_proba() <theanets.feedforward.Classifier.predict_proba>`:

    >>> model.predict_proba(test).shape
    (3, 100, 50)

    Recurrent classifiers have a :func:`predict_sequence` helper method that
    predicts values in an ongoing sequence. Given a seed value, the model
    predicts one time step ahead, then adds the prediction to the seed, predicts
    one more step ahead, and so on:

    >>> seed = np.random.randint(50, size=10).astype('i')
    >>> print(model.predict_sequence(seed, 100))

    See :class:`Text` for more utility code that is helpful for working with
    sequences of class labels.

    See also :ref:`guide-using` for more information.

    Notes
    -----

    Classifier models default to a :class:`cross-entropy
    <theanets.losses.CrossEntropy>` loss. To use a different loss, provide a
    non-default argument for the ``loss`` keyword argument when constructing
    your model.
    '''

    INPUT_NDIM = 3
    '''Number of dimensions for holding input data arrays.'''

    OUTPUT_NDIM = 2
    '''Number of dimensions for holding output data arrays.'''

    def predict_sequence(self, labels, steps, streams=1, rng=None):
        '''Draw a sequential sample of class labels from this network.

        Parameters
        ----------
        labels : list of int
            A list of integer class labels to get the classifier started.
        steps : int
            The number of time steps to sample.
        streams : int, optional
            Number of parallel streams to sample from the model. Defaults to 1.
        rng : :class:`numpy.random.RandomState` or int, optional
            A random number generator, or an integer seed for a random number
            generator. If not provided, the random number generator will be
            created with an automatically chosen seed.

        Yields
        ------
        label(s) : int or list of int
            Yields at each time step an integer class label sampled sequentially
            from the model. If the number of requested streams is greater than
            1, this will be a list containing the corresponding number of class
            labels.
        '''
        if rng is None or isinstance(rng, int):
            rng = np.random.RandomState(rng)
        offset = len(labels)
        batch = max(2, streams)
        inputs = np.zeros((batch, offset + steps, self.layers[0].size), 'f')
        inputs[:, np.arange(offset), labels] = 1
        for i in range(offset, offset + steps):
            chars = []
            for pdf in self.predict_proba(inputs[:i])[:, -1]:
                try:
                    c = rng.multinomial(1, pdf).argmax(axis=-1)
                except ValueError:
                    # sometimes the pdf triggers a normalization error. just
                    # choose greedily in this case.
                    c = pdf.argmax(axis=-1)
                chars.append(int(c))
            inputs[np.arange(batch), i, chars] = 1
            yield chars[0] if streams == 1 else chars

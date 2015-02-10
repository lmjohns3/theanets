# -*- coding: utf-8 -*-

r'''This module contains a class for handling batched datasets.

In most neural network models, parameters must be updated by
:mod:`optimizing <theanets.trainer>` them with respect to a loss function. The
loss function is typically estimated using a set of data that we have gathered
from the problem at hand.

If the problem at hand is "supervised" like a
:class:`classification <theanets.feedforward.Classifier>` or
:class:`regression <theanets.feedforward.Regressor>` problem, then the data to
train your model will require two parts: input samples---for example, photos or
audio recordings---along with target output labels---for example, object classes
that are present in the photos, or words that are present in the audio
recordings. If the problem at hand is "unsupervised" like using an
:class:`autoencoder <theanets.feedforward.Autoencoder>` to learn efficient
representations of data, then your dataset will only require the input samples.

.. note::

    In most mathematics treatments, samples are usually treated as column
    vectors. However, in the ``theanets`` library (as well as many other
    Python-based machine learning libraries), individual samples are treated as
    rows. To avoid confusion with the coding world, the math and code in the
    ``theanets`` documentation assumes row vectors and row-oriented matrices.
'''

import climate
import collections
import numpy.random as rng

logging = climate.get_logger(__name__)


class Dataset:
    '''This class handles batching and shuffling a dataset.

    In ``theanets``, models are :mod:`trained <theanets.trainer>` using sets of
    data collected from the problem at hand; for example, to train an classifier
    for MNIST digits, a labeled training dataset needs to be obtained containing
    a sequence of sample MNIST digit images, and a matched sequence of labels,
    one for each digit.

    During training, data are grouped into "mini-batches"---that is, chunks that
    are larger than 1 sample and smaller than the entire set of samples;
    typically the size of a mini-batch is between 10 and 100, but the specific
    setting can be varied depending on your model, hardware, dataset, and so
    forth. These mini-batches must be presented to the optimization algorithm in
    pseudo-random order to match the underlying stochasticity assumptions of
    many optimization algorithms. This class handles the process of grouping
    data into mini-batches as well as iterating and shuffling these mini-batches
    dynamically as the dataset is consumed by the training algorithm.

    For many tasks, a dataset is obtained as a large block of sample data, which
    in Python is normally assembled as a ``numpy`` ndarray. To use this class on
    such a dataset, just pass in a ``numpy`` array. If labels are required for
    your task, pass a second ``numpy`` array of label data; the two arrays
    should have the same size along their first axis.

    There are some cases (especially when training recurrent networks) when a
    suitable set of training data would be prohibitively expensive to assemble
    in memory as a single ``numpy`` array. To handle these cases, this class can
    also handle a source dataset that is provided via a Python callable. For
    more information on using callables to provide data to your model, see
    :ref:`training-using-callables`.

    Parameters
    ----------
    samples : ndarray or callable
        A set of samples from some data distribution.

        If this parameter is not callable, it is expected to be an ndarray
        containing the "unlabeled" sample data to be used during training,
        validation, etc.

        If this parameter is callable, then mini-batches will be obtained by
        calling the callable with no arguments; the callable is expected to
        return a tuple of ndarrays that will be suitable for training a network.

    labels : ndarray, optional
        A set of labels corresponding to the sample data. The labels array, if
        present, is expected to have the same number of elements along the
        splitting axis as the samples array. This parameter is ignored if
        `samples` is callable.

    name : str, optional
        A string that is used to describe this dataset. Usually something like
        'test' or 'train'.

    batch_size : int, optional
        The size of the mini-batches to create from the data sequences. Defaults
        to 32.

    iteration_size : int, optional
        The number of batches to yield for each call to iterate(). Defaults to
        the length of the data divided by batch_size. If the dataset is a
        callable, then the number is len(callable). If callable has no length,
        then the number is set to 100.

    axis : int, optional
        The axis along which to split the samples and labels. If not provided,
        defaults to 0 (first axis) for 2-dimensional datasets, and to 1 (second
        axis) for 3-dimensional datasets (e.g., for recurrent networks).
    '''

    def __init__(self, samples, labels=None, name=None, batch_size=32,
                 iteration_size=None, axis=None):
        '''Create a minibatch dataset from data arrays or a callable.'''
        self.name = name or 'dataset'
        self.batch_size = batch_size
        self.iteration_size = iteration_size

        self.batches = []

        if isinstance(samples, collections.Callable):
            self._init_callable(samples)
        else:
            self._init_arrays(samples, labels, axis)

    @property
    def number_batches(self):
        return self.iteration_size  # for HF compatibility

    def _init_callable(self, samples):
        self.batches = samples
        if not self.iteration_size:
            try:
                self.iteration_size = len(samples)
            except TypeError: # has no len
                self.iteration_size = 100
        logging.info('%s: %d mini-batches from callable',
                     self.name, self.iteration_size)

    def _init_arrays(self, samples, labels, axis):
        self._index = 0  # index for iteration.

        if axis is None:
            axis = 1 if len(samples.shape) == 3 else 0
        slices = [slice(None), slice(None)][:axis + 1]
        for i in range(0, samples.shape[axis], self.batch_size):
            slices[axis] = slice(i, i + self.batch_size)
            batch = [samples[tuple(slices)]]
            if labels is not None:
                batch.append(labels[tuple(slices)])
            self.batches.append(batch)
        self.shuffle()

        if not self.iteration_size:
            self.iteration_size = len(self.batches)

        shapes = str(self.batches[0][0].shape)
        if labels is not None:
            x, y = self.batches[0]
            shapes = '{} -> {}'.format(x.shape, y.shape)
        logging.info('%s: %d of %d mini-batches of %s',
                     self.name, self.iteration_size,
                     len(self.batches), shapes)

    def __iter__(self):
        return self.iterate(True)

    def shuffle(self):
        rng.shuffle(self.batches)

    def iterate(self, update=True):
        return self._iter_callable() \
            if callable(self.batches) \
            else self._iter_batches(update)

    def _iter_batches(self, update=True):
        k = len(self.batches)
        for _ in range(self.iteration_size):
            self._index += 1
            yield self.batches[self._index % k]
        if update:
            self.update()

    def _iter_callable(self):
        for _ in range(self.iteration_size):
            yield self.batches()

    def update(self):
        if self._index >= len(self.batches):
            self.shuffle()
            self._index = 0

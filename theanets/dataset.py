# Copyright (c) 2012 Leif Johnson <leif@leifjohnson.net>
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

'''This file contains a class for handling batched datasets.'''

import climate
import collections
import numpy.random as rng

logging = climate.get_logger(__name__)


class SequenceDataset:
    '''This class handles batching and shuffling a dataset.

    It's mostly copied from the dataset class from hf.py, except that the
    constructor has slightly different semantics.

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
        callable, then the number is len(callable). If callable has no len, 
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


Dataset = SequenceDataset

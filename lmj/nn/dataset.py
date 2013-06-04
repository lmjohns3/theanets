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

import lmj.cli
import numpy.random as rng

logging = lmj.cli.get_logger(__name__)


class SequenceDataset(object):
    '''This class handles batching and shuffling a dataset.

    It's mostly copied from the dataset class from hf.py, except that the
    constructor has slightly different semantics.
    '''

    def __init__(self, label, *data, **kwargs):
        '''Create a minibatch dataset from a number of different data arrays.

        Arguments:

        label: A string that is used to describe this dataset. Usually something
          like 'test' or 'train'.

        Positional arguments:

        There should be one unnamed keyword argument for each input in the
        neural network that will be processing this dataset. For instance, if
        you are dealing with a classifier network, you'll need one argument for
        the inputs (e.g., mnist digit pixels), and another argument for the
        target outputs (e.g., digit class labels). The order of the arguments
        should be the same as the order of inputs in the network. All arguments
        are expected to have the same number of elements along the first axis.

        Keyword arguments:

        size or batch_size: The size of the mini-batches to create from the
          data matrices. Defaults to 10.
        batches: The number of batches to yield for each call to iterate().
          Defaults to the length of the data divided by batch_size.
        '''
        self.label = label

        n = kwargs.get('size', kwargs.get('batch_size', 10))
        self.minibatches = [
            [d[i:i + n] for d in data] for i in xrange(0, len(data[0]), n)]
        if n == 1:
            self.minibatches = [
                [d[i] for d in data] for i in xrange(len(data[0]))]

        logging.info('data %s: %d mini-batches of %s', label,
                     len(self.minibatches),
                     ', '.join(str(x.shape) for x in self.minibatches[0]))

        self.current = 0
        self.limit = kwargs.get('batches') or len(self.minibatches)
        self.shuffle()

    def __iter__(self):
        return self.iterate(True)

    @property
    def number_batches(self):
        return self.limit

    def shuffle(self):
        rng.shuffle(self.minibatches)

    def iterate(self, update=True):
        k = len(self.minibatches)
        for b in xrange(self.limit):
            yield self.minibatches[(self.current + b) % k]
        if update:
            self.update()

    def update(self):
        if self.current + self.limit >= len(self.minibatches):
            self.shuffle()
            self.current = 0
        else:
            self.current += self.limit

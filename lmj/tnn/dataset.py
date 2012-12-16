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

import logging
import numpy.random as rng


class Dataset(object):
    '''This class handles batching and shuffling a dataset.

    It's heavily inspired by the dataset class from hf.py.
    '''

    def __init__(self, label, *data, **kwargs):
        self.label = label

        n = kwargs.get('size', kwargs.get('batch_size', 100))
        self.batches = [
            [d[i:i + n] for d in data] for i in xrange(0, len(data[0]), n)]

        d = self.batches
        shape = []
        while True:
            try:
                shape.append(len(d))
                d = d[0]
            except:
                break
        logging.info('data %s: %s', label, shape)

        self.current = 0
        self.limit = kwargs.get('batches') or len(self.batches)
        self.shuffle()

    @property
    def number_batches(self):
        return self.limit

    def shuffle(self):
        rng.shuffle(self.batches)

    def __iter__(self):
        return self.iterate(True)

    def iterate(self, update=True):
        k = len(self.batches)
        for b in xrange(self.limit):
            yield self.batches[(self.current + b) % k]
        if update:
            self.update()

    def update(self):
        if self.current + self.limit >= len(self.batches):
            self.shuffle()
            self.current = 0
        else:
            self.current += self.limit

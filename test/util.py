'''Helper code for theanets unit tests.'''

import numpy as np
import skdata.mnist


class MNIST(object):
    NUM_DIGITS = 100
    DIGIT_SIZE = 784

    def setUp(self):
        np.random.seed(3)

        mnist = skdata.mnist.dataset.MNIST()
        mnist.meta  # trigger download if needed.
        def arr(n, dtype):
            arr = mnist.arrays[n]
            return arr.reshape((len(arr), -1)).astype(dtype)
        self.images = arr('train_images', 'f')[:MNIST.NUM_DIGITS] / 255.
        self.labels = arr('train_labels', 'b')[:MNIST.NUM_DIGITS]

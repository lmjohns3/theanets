'''Helper code for theanets unit tests.'''

import numpy as np


class MNIST(object):
    NUM_DIGITS = 100
    DIGIT_SIZE = 784

    def setUp(self):
        # we just create some random "mnist digit" data of the right shape.
        np.random.seed(3)
        self.images = np.random.randn(MNIST.NUM_DIGITS, MNIST.DIGIT_SIZE).astype('f')
        self.labels = np.random.randint(0, 10, MNIST.NUM_DIGITS).astype('i')

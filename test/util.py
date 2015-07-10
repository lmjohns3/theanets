'''Helper code for theanets unit tests.'''

import numpy as np


class Base(object):
    NUM_EXAMPLES = 64
    NUM_INPUTS = 7
    NUM_OUTPUTS = 3
    NUM_CLASSES = 5

    INPUTS = np.random.randn(NUM_EXAMPLES, NUM_INPUTS).astype('f')
    INPUT_WEIGHTS = abs(np.random.randn(NUM_EXAMPLES, NUM_INPUTS)).astype('f')
    OUTPUTS = np.random.randn(NUM_EXAMPLES, NUM_OUTPUTS).astype('f')
    OUTPUT_WEIGHTS = abs(np.random.randn(NUM_EXAMPLES, NUM_OUTPUTS)).astype('f')
    CLASSES = np.random.randint(NUM_CLASSES, size=NUM_EXAMPLES).astype('i')
    CLASS_WEIGHTS = abs(np.random.rand(NUM_EXAMPLES)).astype('f')

    def assert_progress(self, algo, data, **kwargs):
        trainer = self.exp.itertrain(
            data, algorithm=algo, monitor_gradients=True, batch_size=3, **kwargs)
        train0, valid0 = next(trainer)
        train1, valid1 = next(trainer)
        assert train1['loss'] < valid0['loss']   # should have made progress!
        assert valid1['loss'] == valid0['loss']  # no new validation occurred

    def assert_shape(self, actual, expected):
        if not isinstance(expected, tuple):
            expected = (self.NUM_EXAMPLES, expected)
        assert actual == expected, 'expected {}, got {}'.format(expected, actual)


class RecurrentBase(Base):
    NUM_TIMES = 31
    NUM_EXAMPLES = Base.NUM_EXAMPLES
    NUM_INPUTS = Base.NUM_INPUTS
    NUM_OUTPUTS = Base.NUM_OUTPUTS
    NUM_CLASSES = Base.NUM_CLASSES

    INPUTS = np.random.randn(NUM_EXAMPLES, NUM_TIMES, NUM_INPUTS).astype('f')
    INPUT_WEIGHTS = abs(
        np.random.randn(NUM_EXAMPLES, NUM_TIMES, NUM_INPUTS)).astype('f')
    OUTPUTS = np.random.randn(NUM_EXAMPLES, NUM_TIMES, NUM_OUTPUTS).astype('f')
    OUTPUT_WEIGHTS = abs(
        np.random.randn(NUM_EXAMPLES, NUM_TIMES, NUM_OUTPUTS)).astype('f')
    CLASSES = np.random.randn(NUM_EXAMPLES, NUM_TIMES).astype('i')
    CLASS_WEIGHTS = abs(np.random.rand(NUM_EXAMPLES, NUM_TIMES)).astype('f')

    def assert_shape(self, actual, expected):
        if not isinstance(expected, tuple):
            expected = (self.NUM_EXAMPLES, self.NUM_TIMES, expected)
        assert actual == expected, 'expected {}, got {}'.format(expected, actual)

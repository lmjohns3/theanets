'''Helper code for theanets unit tests.'''

import climate
import numpy as np

climate.enable_default_logging()

np.random.seed(13)

NUM_EXAMPLES = 64
NUM_INPUTS = 7
NUM_HID1 = 8
NUM_HID2 = 12
NUM_OUTPUTS = 5
NUM_CLASSES = 6

INPUTS = np.random.randn(NUM_EXAMPLES, NUM_INPUTS).astype('f')
INPUT_WEIGHTS = abs(np.random.randn(NUM_EXAMPLES, NUM_INPUTS)).astype('f')
OUTPUTS = np.random.randn(NUM_EXAMPLES, NUM_OUTPUTS).astype('f')
OUTPUT_WEIGHTS = abs(np.random.randn(NUM_EXAMPLES, NUM_OUTPUTS)).astype('f')
CLASSES = np.random.randint(NUM_CLASSES, size=NUM_EXAMPLES).astype('i')
CLASS_WEIGHTS = abs(np.random.rand(NUM_EXAMPLES)).astype('f')

AE_DATA = [INPUTS]
WAE_DATA = [INPUTS, INPUT_WEIGHTS]
AE_LAYERS = [NUM_INPUTS, NUM_HID1, NUM_HID2, NUM_INPUTS]

CLF_DATA = [INPUTS, CLASSES]
WCLF_DATA = [INPUTS, CLASSES, CLASS_WEIGHTS]
CLF_LAYERS = [NUM_INPUTS, NUM_HID1, NUM_HID2, NUM_CLASSES]

REG_DATA = [INPUTS, OUTPUTS]
WREG_DATA = [INPUTS, OUTPUTS, OUTPUT_WEIGHTS]
REG_LAYERS = [NUM_INPUTS, NUM_HID1, NUM_HID2, NUM_OUTPUTS]


class RNN:
    NUM_TIMES = 31

    INPUTS = np.random.randn(NUM_EXAMPLES, NUM_TIMES, NUM_INPUTS).astype('f')
    INPUT_WEIGHTS = abs(
        np.random.randn(NUM_EXAMPLES, NUM_TIMES, NUM_INPUTS)).astype('f')
    OUTPUTS = np.random.randn(NUM_EXAMPLES, NUM_TIMES, NUM_OUTPUTS).astype('f')
    OUTPUT_WEIGHTS = abs(
        np.random.randn(NUM_EXAMPLES, NUM_TIMES, NUM_OUTPUTS)).astype('f')
    CLASSES = np.random.randn(NUM_EXAMPLES, NUM_TIMES).astype('i')
    CLASS_WEIGHTS = abs(np.random.rand(NUM_EXAMPLES, NUM_TIMES)).astype('f')

    AE_DATA = [INPUTS]
    WAE_DATA = [INPUTS, INPUT_WEIGHTS]

    CLF_DATA = [INPUTS, CLASSES]
    WCLF_DATA = [INPUTS, CLASSES, CLASS_WEIGHTS]

    REG_DATA = [INPUTS, OUTPUTS]
    WREG_DATA = [INPUTS, OUTPUTS, OUTPUT_WEIGHTS]


class CNN:
    NUM_WIDTH = 13
    NUM_HEIGHT = 15

    FILTER_WIDTH = 4
    FILTER_HEIGHT = 3
    FILTER_SIZE = (FILTER_WIDTH, FILTER_HEIGHT)

    INPUTS = np.random.randn(
        NUM_EXAMPLES, NUM_WIDTH, NUM_HEIGHT, NUM_INPUTS).astype('f')

    CLF_DATA = [INPUTS, CLASSES]
    WCLF_DATA = [INPUTS, CLASSES, CLASS_WEIGHTS]

    REG_DATA = [INPUTS, OUTPUTS]
    WREG_DATA = [INPUTS, OUTPUTS, OUTPUT_WEIGHTS]


def assert_progress(model, data, algo='sgd'):
    trainer = model.itertrain(
        data, algo=algo, momentum=0.5, batch_size=3, max_gradient_norm=1)
    train0, valid0 = next(trainer)
    train1, valid1 = next(trainer)
    assert train1['loss'] < valid0['loss']   # should have made progress!
    assert valid1['loss'] == valid0['loss']  # no new validation occurred


def assert_shape(actual, expected):
    if not isinstance(expected, tuple):
        expected = (NUM_EXAMPLES, expected)
    assert actual == expected

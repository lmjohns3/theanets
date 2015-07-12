#!/usr/bin/env python

'''Theanets example using a deep bidirectional LSTM for phoneme classification.

This example loads an audio classification benchmark from github, defines a
callable for extracting batches from the downloaded dataset, and trains a deep
classifier network on the data. The network that is evaluated as part of the
benchmarks is a three-layer bidirectional LSTM. Typically the model exceeds 90%
accuracy on the training set, but reaches only about 70% accuracy on the
validation set. Clearly overtraining is a critical issue here.

This example only works with Python 2 at the moment.
'''

import climate
import io
import numpy as np
import theanets
import scipy.io
import os
import tempfile
import urllib
import zipfile

logging = climate.get_logger('lstm-chime')

climate.enable_default_logging()

BATCH_SIZE = 32
TRAIN_NC = os.path.join(tempfile.gettempdir(), 'chime1_train.nc')
VALID_NC = os.path.join(tempfile.gettempdir(), 'chime1_valid.nc')
ZIPURL = 'https://github.com/craffel/lstm_benchmarks/archive/master.zip'

# If needed, get the data files from https://github.com/craffel/lstm_benchmarks.
if not os.path.isfile(TRAIN_NC) or not os.path.isfile(VALID_NC):
    logging.info('attempting data copy from url: %s', ZIPURL)
    z = zipfile.ZipFile(io.BytesIO(urllib.urlopen(ZIPURL).read()))
    with open(TRAIN_NC, 'wb') as savefile:
        savefile.write(z.read('lstm_benchmarks-master/data/train_1_speaker.nc'))
    with open(VALID_NC, 'wb') as savefile:
        savefile.write(z.read('lstm_benchmarks-master/data/val_1_speaker.nc'))
    z.close()


def batch_at(features, labels, seq_begins, seq_lengths):
    '''Extract a single batch of data to pass to the model being trained.

    Parameters
    ----------
    features, labels : ndarray
        Arrays of the input features and target labels.
    seq_begins : ndarray
        Array of the start offsets of the speech segments to include.
    seq_lengths : ndarray
        Array of the lengths of the speech segments to include in the batch.

    Returns
    -------
    features, labels, mask : ndarrays
        A triple of arrays for training a network. The first element contains
        input features, the second contains target labels, and the third
        contains a "mask" consisting of ones where there is valid data and zeros
        everywhere else.
    '''
    length = seq_lengths.max()
    feat = np.zeros((BATCH_SIZE, length, features.shape[-1]), 'f')
    labl = np.zeros((BATCH_SIZE, length), 'i')
    mask = np.zeros((BATCH_SIZE, length), 'f')
    for b, (begin, length) in enumerate(zip(seq_begins, seq_lengths)):
        feat[b, :length] = features[begin:begin+length]
        labl[b, :length] = labels[begin:begin+length]
        mask[b, :length] = 1
    return [feat, labl, mask]


def batches(dataset):
    '''Returns a callable that chooses sequences from netcdf data.'''
    seq_lengths = dataset.variables['seqLengths'].data
    seq_begins = np.concatenate(([0], np.cumsum(seq_lengths)[:-1]))

    def sample():
        chosen = np.random.choice(
            list(range(len(seq_lengths))), BATCH_SIZE, replace=False)
        return batch_at(dataset.variables['inputs'].data,
                        dataset.variables['targetClasses'].data,
                        seq_begins[chosen],
                        seq_lengths[chosen])

    return sample


# Now that we can load data, we construct a recurrent classifier model and then
# train it up! Training progress will be displayed on the console. This example
# can take a good while to run, especially the first time it is run (it takes
# about 20min to compile the model from scratch, but only a few minutes if all
# of the compiler targets are cached).

def layer(n):
    '''Helper for building a bidirectional LSTM layer with n cells.'''
    return dict(form='bidirectional', worker='lstm', size=n)

n = theanets.recurrent.Classifier(
    layers=(39, layer(156), layer(300), layer(102), (51, 'softmax')),
    weighted=True,
)

n.train(
    batches(scipy.io.netcdf_file(TRAIN_NC)),
    batches(scipy.io.netcdf_file(VALID_NC)),
    algo='rmsprop',
    learning_rate=0.0001,
    momentum=0.9,
    max_gradient_clip=1,
    input_noise=0.6,
    train_batches=30,
    valid_batches=3,
    batch_size=BATCH_SIZE,
)

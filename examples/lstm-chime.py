#!/usr/bin/env python

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

if not os.path.isfile(TRAIN_NC) or not os.path.isfile(VALID_NC):
    # get the data files from the repository at https://github.com/craffel/lstm_benchmarks
    zipurl = 'https://github.com/craffel/lstm_benchmarks/archive/master.zip'
    logging.info('attempting data copy from url: %s', zipurl)
    z = zipfile.ZipFile(io.BytesIO(urllib.urlopen(zipurl).read()))
    with open(TRAIN_NC, 'wb') as savefile:
        savefile.write(z.read('lstm_benchmarks-master/data/train_1_speaker.nc'))
    with open(VALID_NC, 'wb') as savefile:
        savefile.write(z.read('lstm_benchmarks-master/data/val_1_speaker.nc'))
    z.close()


def batch_at(features, labels, seq_begins, seq_lengths):
    length = seq_lengths.max()
    feat = np.zeros((length, BATCH_SIZE, features.shape[-1]), 'f')
    labl = np.zeros((length, BATCH_SIZE), 'int32')
    mask = np.zeros((length, BATCH_SIZE), 'f')
    for b, (begin, length) in enumerate(zip(seq_begins, seq_lengths)):
        feat[:length, b] = features[begin:begin+length]
        labl[:length, b] = labels[begin:begin+length]
        mask[:length, b] = 1
    return [feat, labl, mask]

# returns a callable that chooses sequences from netcdf data
def batches(dataset):
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


def layer(n):
    return dict(form='bidirectional', worker='lstm', size=n)

e = theanets.Experiment(
    theanets.recurrent.Classifier,
    layers=(39, layer(156), layer(300), layer(102), (51, 'softmax')),
    weighted=True,
)

e.train(
    batches(scipy.io.netcdf_file(TRAIN_NC)),
    batches(scipy.io.netcdf_file(VALID_NC)),
    algorithm='rmsprop',
    learning_rate=0.0001,
    max_gradient_norm=1,
    input_noise=0.6,
    train_batches=30,
    valid_batches=3,
    batch_size=BATCH_SIZE,
)

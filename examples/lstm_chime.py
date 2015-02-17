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

# do fixed segments for now (warning: each segment does not correspond to real utterances!)
SEQLEN = 100
BATCH_SIZE = 50
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

def batch_at(data, start):
    dtype, shape = 'int32', (SEQLEN, BATCH_SIZE)
    if len(data.shape) > 1:
        dtype, shape = 'f', (SEQLEN, BATCH_SIZE, data.shape[-1])
    X = np.zeros(shape, dtype=dtype)
    for b in range(BATCH_SIZE):
        s = start + SEQLEN * b
        X[:, b] = data[s:s + SEQLEN]
    return X

def batches(dataset):
    steps = dataset.dimensions['numTimesteps']
    def sample():
        i = np.random.randint(steps - SEQLEN * BATCH_SIZE)
        s = batch_at(dataset.variables['inputs'].data, i)
        t = batch_at(dataset.variables['targetClasses'].data, i)
        return [s, t]
    return sample

def layer(n):
    return dict(form='bidirectional', worker='lstm', size=n)

e = theanets.Experiment(
    theanets.recurrent.Classifier,
    layers=(39, layer(156), layer(300), layer(102), 51),
    recurrent_error_start=0,
    batch_size=BATCH_SIZE,
    input_noise=0.6,
    max_gradient_norm=10,
)
e.train(batches(scipy.io.netcdf_file(TRAIN_NC)),
        batches(scipy.io.netcdf_file(VALID_NC)))

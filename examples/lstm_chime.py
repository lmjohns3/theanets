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
    max_seq_length = np.max(seq_lengths)
    batch_size = len(seq_begins)
    ftype, fshape = 'f', (max_seq_length, batch_size, features.shape[-1])
    ltype, lshape = 'int32', (max_seq_length, batch_size)
    mtype, mshape = 'f', (max_seq_length, batch_size)
    feat = np.zeros(fshape, dtype=ftype)
    labl = np.zeros(lshape, dtype=ltype)
    mask = np.zeros(mshape, dtype=mtype)
    for b, (begin, length) in enumerate(zip(seq_begins, seq_lengths)):
        feat[:length, b] = features[begin:begin+length]
        labl[:length, b] = labels[begin:begin+length]
        mask[:length, b] = np.ones(length)
    return [feat, labl, mask]

# returns a callable that chooses sequences from netcdf data
# the callable (sample) does random sequence shuffling without replacement
# or can get deterministic ordered batches
# circles back to choose from all the sequences once the unchosen set becomes zero size
def batches(dataset, choose_random=True):
    steps = dataset.dimensions['numTimesteps']
    seq_lengths = dataset.variables['seqLengths'].data
    seq_begins = np.concatenate(([0], np.cumsum(seq_lengths)[:-1]))
    state = dict(unchosen=set(range(len(seq_lengths))))
    def sample():
        unchosen = state['unchosen']
        if len(unchosen) < BATCH_SIZE:
            unchosen = set(range(len(seq_lengths)))
        if choose_random:
            chosen = np.random.choice(list(unchosen), BATCH_SIZE, replace=False)
        else:
            chosen = list(unchosen)[:BATCH_SIZE]
        state['unchosen'] = unchosen - set(chosen)
        return batch_at(dataset.variables['inputs'].data,
                        dataset.variables['targetClasses'].data,
                        seq_begins[chosen],
                        seq_lengths[chosen])
    return sample


def layer(n):
    return dict(form='bidirectional', worker='lstm', size=n)

e = theanets.Experiment(
    theanets.recurrent.Classifier,
    layers=(39, layer(156), layer(300), layer(102), 51),
    batch_size=BATCH_SIZE,
    input_noise=0.6,
    max_gradient_norm=10,
    weighted=True,
)
e.train(batches(scipy.io.netcdf_file(TRAIN_NC)),
        batches(scipy.io.netcdf_file(VALID_NC)))

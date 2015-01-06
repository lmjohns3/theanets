#!/usr/bin/env python

import climate
import matplotlib.pyplot as plt
import numpy as np
import theanets
import scipy.io
import os.path

from StringIO import StringIO
from zipfile import ZipFile
from urllib import urlopen

climate.enable_default_logging()
logging = climate.get_logger('lstm_chime_example')

# do fixed segments for now (warning: each segment does not correspond to real utterances!)
SEQLEN = 100
BATCH_SIZE = 50

BATCH_STEP = BATCH_SIZE * SEQLEN

TRAIN_NC = '.chime1_data/train_1_speaker.nc'
VAL_NC = '.chime1_data/val_1_speaker.nc'

if not os.path.isfile(TRAIN_NC) or not os.path.isfile(VAL_NC):
    if not os.path.exists(os.path.dirname(TRAIN_NC)):
        os.makedirs(os.path.dirname(TRAIN_NC))
    # get the data files from the repository at https://github.com/craffel/lstm_benchmarks
    zipurl="https://github.com/craffel/lstm_benchmarks/archive/master.zip"
    logging.info('attempting data copy from url: %s', zipurl)
    url = urlopen(zipurl)
    z = ZipFile(StringIO(url.read()))
    data=z.read('lstm_benchmarks-master/data/train_1_speaker.nc')
    savefile = open(TRAIN_NC, "wb")
    savefile.write(data)
    savefile.close()
    data=z.read('lstm_benchmarks-master/data/val_1_speaker.nc')
    savefile = open(VAL_NC, "wb")
    savefile.write(data)
    savefile.close()
    data=None
    z.close()
    logging.info("Done (hopefully).")

with open(TRAIN_NC, 'r') as f:
    train_data = scipy.io.netcdf_file(f)
with open(VAL_NC, 'r') as f:
    val_data = scipy.io.netcdf_file(f)

trainV = train_data.variables
trainD = train_data.dimensions

valV = val_data.variables
valD = val_data.dimensions

nbatch_train = np.int_(np.floor(trainD['numTimesteps']/BATCH_STEP))
nbatch_val = np.int_(np.floor(valD['numTimesteps']/BATCH_STEP))

XDIM = trainD['inputPattSize']
NCLASS = trainD['numLabels']

# simple lstm with one hidden layer (for now)
e = theanets.Experiment(
    theanets.recurrent.Classifier,
    layers=(39, 78, 51),
    recurrent_error_start=0,
    recurrent_form='LSTM',
    output_activation='softmax',
    hidden_activation='tanh',
    input_noise=0.5,
    batch_size=BATCH_SIZE,
    train_batches=nbatch_train)

def get_batch(V, variable, batch, dim, seqlen=SEQLEN, batch_size=BATCH_SIZE):
    '''
        Read from netcdf file V
        
        :parameters:
        - V : scipy.io.netcdf_file.variables object
        scipy.io.netcdf_file.variables object to read variables from the netcdf file
        - variable : string
        variable name in the netcdf file
        - batch : int
        which batch to use 
        - dim : int
        feature dimension
        - seqlen : int
        sequence length
        - batch_size : int
        Mini-batch size
        
        :returns:
        - X_batch : np.ndarray
        Tensor of time series matrix batches,
        shape=(sequence_length, batch_size, dim) or shape=(sequence_length, batch_size) if dim=1
        '''
    # returns whole data
    if (seqlen == -1):
        X_batch = V[variable].data.astype('f')
    else:  # returns 3D fixed size batch data
        if (dim > 1):
            X_batch = np.zeros((seqlen, batch_size, dim), dtype='f')
        else:
            X_batch = np.zeros((seqlen, batch_size), dtype='int32')
        for n in range(batch_size):
            for t in range(seqlen):
                if (dim > 1):
                    X_batch[t, n, 0:dim] = V[variable].data[batch*(batch_size*seqlen)+n*seqlen+t,0:dim].astype('f')
                else:
                    X_batch[t, n] = V[variable].data[batch*(batch_size*seqlen)+n*seqlen+t].astype('int32')
    return X_batch

# each time, we read data from the netcdf file. 
# I want the code to be generalizable to larger size files
# which may not fit in the memory, 
# so I avoid reading the whole file at once
def generate():
    i = np.random.choice(nbatch_train)  # random batches for now, should change to sample without replacement later
    s = get_batch(trainV,'inputs',i,XDIM)
    t = get_batch(trainV,'targetClasses',i,1)
    return [s, t]

X_vals=np.zeros((nbatch_val, SEQLEN, BATCH_SIZE, XDIM), dtype='f')
y_vals=np.zeros((nbatch_val, SEQLEN, BATCH_SIZE ), dtype='int32')

for i in range(nbatch_val):
    X_vals[i]=get_batch(valV,'inputs',i,XDIM)
    y_vals[i]=get_batch(valV,'targetClasses',i,1)

# get a random batch to check sizes
[tr_x,tr_y]= generate()

logging.info('train data each batch: %s -> %s', tr_x.shape, tr_y.shape)
logging.info('validation data all batches: %s -> %s', X_vals.shape, y_vals.shape)

# evaluate performance on the first batch of the validation data only (for now)
e.train(generate, (X_vals[0], y_vals[0]), momentum=0.99)
#e.train(generate, momentum=0.99)


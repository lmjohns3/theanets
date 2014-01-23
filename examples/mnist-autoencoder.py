#!/usr/bin/env python

import climate
import cPickle
import gzip
import logging
import matplotlib.pyplot as plt
import numpy as np
import os
import tempfile
import theanets
import urllib

from plot_utils import plot_autoencoder_experiment

climate.enable_default_logging()

URL = 'http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz'
DATASET = os.path.join(tempfile.gettempdir(), 'mnist.pkl.gz')

if not os.path.isfile(DATASET):
    logging.info('downloading mnist digit dataset from %s' % URL)
    urllib.urlretrieve(URL, DATASET)
    logging.info('saved mnist digits to %s' % DATASET)

train, valid, _ = [x for x, _ in cPickle.load(gzip.open(DATASET))]
N = 16
e = theanets.Experiment(theanets.Autoencoder, layers=(784, N * N, 784))
e.run(train, valid)
plot_autoencoder_experiment(e, valid)

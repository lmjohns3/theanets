#!/usr/bin/env python
import cPickle
import gzip
import logging
import lmj.cli
import theanets
import matplotlib.pyplot as plt
import numpy as np
import os
import tempfile
import urllib
from plot_utils import plot_autoencoder_experiment

lmj.cli.enable_default_logging()

URL = 'http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz'
DATASET = os.path.join(tempfile.gettempdir(), 'mnist.pkl.gz')

if not os.path.isfile(DATASET):
    logging.info('downloading mnist digit dataset from %s' % URL)
    urllib.urlretrieve(URL, DATASET)
    logging.info('saved mnist digits to %s' % DATASET)

train, valid, _ = [x for x, _ in cPickle.load(gzip.open(DATASET))]
e = theanets.Experiment(theanets.Autoencoder,
                        layers=(784, 250, 150, 30, 150, 250, 784), learning_rate=.005, learning_rate_decay=.1, patience=20, optimize="sgd",
                        num_updates=256,
                        tied_weights=True,
                        batch_size=32,
                        )
e.run(train, valid)

plot_autoencoder_experiment(e, valid)

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
#This example is not currently working
layerwise_kwargs = {'num_updates':15, 'input_dropouts':.2, 'hidden_dropouts':.5}
sgd_kwargs = {'num_updates':500}
e = theanets.Experiment(theanets.Autoencoder,
        optimize=['layerwise','sgd'],
        trainer_specific_args=[layerwise_kwargs,
                               sgd_kwargs],
        tied_weights=True,
        layers=(784,1000,500,250,30,250,500,1000,784))
e.run(train, valid)

plot_autoencoder_experiment(e, valid)

#!/usr/bin/env python

import cPickle
import gzip
import logging
import lmj.cli
import matplotlib.pyplot as plt
import numpy as np
import os
import tempfile
import theanets
import urllib
from plot_utils import plot_classifier_experiment
lmj.cli.enable_default_logging()

URL = 'http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz'
DATASET = os.path.join(tempfile.gettempdir(), 'mnist.pkl.gz')

if not os.path.isfile(DATASET):
    logging.info('downloading mnist digit dataset from %s' % URL)
    urllib.urlretrieve(URL, DATASET)
    logging.info('saved mnist digits to %s' % DATASET)

train, valid, _ = [
    (x, y.astype('int32')) for x, y in cPickle.load(gzip.open(DATASET))]

e = theanets.Experiment(theanets.Classifier, layers=(784, 200, 10))
e.run(train, valid)
plot_classifier_experiment(e, valid)

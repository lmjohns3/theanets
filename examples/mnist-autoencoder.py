#!/usr/bin/env python

import cPickle
import gzip
import logging
import os
import sys
import urllib

import lmj.tnn

logging.basicConfig(
    stream=sys.stdout,
    format='%(levelname).1s %(asctime)s %(message)s',
    level=logging.INFO)

URL = 'http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz'
DATASET = 'mnist.pkl.gz'

if not os.path.isfile(DATASET):
    logging.info('downloading mnist digit dataset from %s' % URL)
    urllib.urlretrieve(URL, DATASET)
    logging.info('saved mnist digits to %s' % DATASET)

class Main(lmj.tnn.Main):
    def get_network(self):
        return lmj.tnn.Autoencoder

    def get_datasets(self):
        return [(x, ) for x, _ in cPickle.load(gzip.open(DATASET))]

Main().train().save('mnist-autoencoder.pkl.gz')

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

read = lambda s: cPickle.load(gzip.open(s))

net = lmj.tnn.main(
    lmj.tnn.Autoencoder,
    lambda *_: [(x, ) for x, _ in read(DATASET)])

net.save('net.pkl.gz')

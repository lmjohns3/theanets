#!/usr/bin/env python

import cPickle
import gzip
import logging
import lmj.cli
import lmj.nn
import os
import tempfile
import urllib

lmj.cli.enable_default_logging()

URL = 'http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz'
DATASET = os.path.join(tempfile.gettempdir(), 'mnist.pkl.gz')

if not os.path.isfile(DATASET):
    logging.info('downloading mnist digit dataset from %s' % URL)
    urllib.urlretrieve(URL, DATASET)
    logging.info('saved mnist digits to %s' % DATASET)

class Main(lmj.nn.Main):
    def get_network(self):
        return lmj.nn.Autoencoder

    def get_datasets(self):
        return [x for x, _ in cPickle.load(gzip.open(DATASET))]

m = Main(layers=(784, 200, 784))
path = os.path.join(
    tempfile.gettempdir(),
    'mnist-autoencoder-%s.pkl.gz' % ','.join(str(n) for n in m.args.layers))
if os.path.exists(path):
    m.net.load(path)
m.train()
m.net.save(path)

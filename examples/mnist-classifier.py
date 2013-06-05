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

train, valid, _ = [
    (x, y.astype('int32')) for x, y in cPickle.load(gzip.open(DATASET))]

e = lmj.nn.Experiment(lmj.nn.Classifier, layers=(784, 200, 10))

path = os.path.join(
    tempfile.gettempdir(),
    'mnist-classifier-%s.pkl.gz' % ','.join(str(n) for n in e.args.layers))

if os.path.exists(path):
    e.load(path)
e.run(train, valid)
e.save(path)

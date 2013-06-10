#!/usr/bin/env python

import cPickle
import gzip
import logging
import lmj.cli
import lmj.nn
import matplotlib.pyplot as plt
import numpy as np
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

train, valid, _ = [x for x, _ in cPickle.load(gzip.open(DATASET))]

e = lmj.nn.Experiment(lmj.nn.Autoencoder, layers=(784, 200, 784))
e.run(train, valid)

# make a plot showing the learned basis functions as 28x28 images.
img = np.zeros((231, 231), float)
for i, d in enumerate(e.network.weights[0].get_value().T):
    if i == 64:
        break
    a, b = divmod(i, 8)
    img[a * 29:a * 29 + 28, b * 29:b * 29 + 28] = d.reshape((28, 28))
ax = plt.subplot(111)
ax.set_frame_on(False)
ax.imshow(img, cmap='gray')
plt.show()

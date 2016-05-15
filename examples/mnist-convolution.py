#!/usr/bin/env python

import matplotlib.pyplot as plt
import theanets

from utils import load_mnist, plot_filters


SHAPE = (28, 28, 1)

train, valid, _ = load_mnist(labels=True)

X, y = train
train = X.reshape((-1, ) + SHAPE), y

X, y = valid
valid = X.reshape((-1, ) + SHAPE), y

net = theanets.convolution.Classifier([
    SHAPE, dict(form='conv2', size=100, filter_size=(14, 14)), 'flatten', 10])
net.train(train, valid, train_batches=100, valid_batches=100)

plot_filters(net.find('hid1', 'w'))
plt.tight_layout()
plt.show()

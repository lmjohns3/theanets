#!/usr/bin/env python

import matplotlib.pyplot as plt
import theanets

from utils import load_mnist, plot_filters


train, valid, _ = load_mnist(labels=True)

X, y = train
train = X.reshape((-1, 28, 28, 1)), y

X, y = valid
valid = X.reshape((-1, 28, 28, 1)), y

N = 10

net = theanets.convolution.Classifier([
    1,
    dict(form='conv2', size=N * N, filter_size=(14, 14)),
    ('flat', 22500),  # 22500 = N * N * (28 - 14 + 1) * (28 - 14 + 1)
    10
])
net.train(train, valid,
          train_batches=100,
          valid_batches=100)

plot_filters(net.find('hid1', 'w'))
plt.tight_layout()
plt.show()

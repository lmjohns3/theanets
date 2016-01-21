#!/usr/bin/env python

import matplotlib.pyplot as plt
import theanets

from utils import load_mnist, plot_layers


train, valid, _ = load_mnist(labels=True)

N = 10

net = theanets.Classifier([784, N * N, ('softmax', 10)])
net.train(train, valid, min_improvement=0.001, train_batches=100)

plot_layers([net.find('hid1', 'w'), net.find('out', 'w')])
plt.tight_layout()
plt.show()

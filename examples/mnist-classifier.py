#!/usr/bin/env python

import matplotlib.pyplot as plt
import theanets

from utils import load_mnist, plot_layers


train, valid, _ = load_mnist(labels=True)

N = 10

e = theanets.Experiment(
    theanets.Classifier,
    layers=(784, N * N, ('softmax', 10)),
)
e.train(train, valid, min_improvement=0.001)

plot_layers([e.network.find('hid1', 'w'), e.network.find('out', 'w')])
plt.tight_layout()
plt.show()

#!/usr/bin/env python

import matplotlib.pyplot as plt
import theanets

from utils import load_mnist, plot_layers, plot_images


e = theanets.Experiment(
    theanets.Classifier,
    layers=(784, 1024, 256, 64, 10),
    train_batches=100,
)

# first, run an unsupervised layerwise pretrainer.
train, valid, _ = load_mnist()
e.train(train, valid, optimize='pretrain', patience=1, min_improvement=0.1)

# second, run a supervised trainer on the classifier model.
train, valid, _ = load_mnist(labels=True)
e.train(train, valid)

plot_layers([e.network.find(i, 0) for i in (1, 2, 3)], tied_weights=True)
plt.tight_layout()
plt.show()

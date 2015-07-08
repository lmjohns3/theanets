#!/usr/bin/env python

import matplotlib.pyplot as plt
import theanets

from utils import load_mnist, plot_layers, plot_images


net = theanets.Classifier(
    layers=(784, 1024, 256, 64, ('softmax', 10)),
)

# first, run an unsupervised layerwise pretrainer.
train, valid, _ = load_mnist()
net.train(train, valid,
          algo='pretrain',
          patience=1,
          min_improvement=0.1,
          train_batches=100)

# second, run a supervised trainer on the classifier model.
train, valid, _ = load_mnist(labels=True)
net.train(train, valid, min_improvement=0.01, train_batches=100)

plot_layers([net.find(i, 'w') for i in (1, 2, 3)])
plt.tight_layout()
plt.show()

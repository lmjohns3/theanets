#!/usr/bin/env python

import matplotlib.pyplot as plt
import theanets

from utils import load_mnist, plot_layers, plot_images


train, valid, _ = load_mnist()

N = 8

e = theanets.Experiment(
    theanets.Autoencoder,
    layers=(784, N * N, 784),
    train_batches=100,
)
e.run(train, valid)

plot_layers(e.network.weights)
plt.tight_layout()
plt.show()

valid = valid[:N*N]
plot_images(valid, 121, 'Sample data')
plot_images(e.network.predict(valid), 122, 'Reconstructed data')
plt.tight_layout()
plt.show()

#!/usr/bin/env python

import matplotlib.pyplot as plt
import theanets

from utils import load_mnist, plot_layers, plot_images


train, valid, _ = load_mnist()

e = theanets.Experiment(
    theanets.Autoencoder,
    layers=(784, 256, 64, 36, 64, 256, 784),
    train_batches=100,
    tied_weights=True,
)
e.run(train, valid)

plot_layers(e.network.weights, tied_weights=True)
plt.tight_layout()
plt.show()

valid = valid[:16*16]
plot_images(valid, 121, 'Sample data')
plot_images(e.network.predict(valid), 122, 'Reconstructed data')
plt.tight_layout()
plt.show()

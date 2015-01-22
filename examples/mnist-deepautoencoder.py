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
e.train(train, valid, optimize='layerwise', patience=1, min_improvement=0.1)
e.train(train, valid)

plot_layers([e.network.get_weights(i) for i in (1, 2, 3)], tied_weights=True)
plt.tight_layout()
plt.show()

valid = valid[:16*16]
plot_images(valid, 121, 'Sample data')
plot_images(e.network.predict(valid), 122, 'Reconstructed data')
plt.tight_layout()
plt.show()

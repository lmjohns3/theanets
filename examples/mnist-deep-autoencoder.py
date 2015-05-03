#!/usr/bin/env python

import matplotlib.pyplot as plt
import theanets

from utils import load_mnist, plot_layers, plot_images


train, valid, _ = load_mnist()

e = theanets.Experiment(
    theanets.Autoencoder,
    layers=(784, 256, 100, 64, ('tied', 100), ('tied', 256), ('tied', 784)),
)
e.train(train, valid,
        algorithm='layerwise',
        patience=1,
        min_improvement=0.05,
        train_batches=100)
e.train(train, valid, min_improvment=0.01, train_batches=100)

plot_layers([e.network.find(i, 'w') for i in (1, 2, 3)], tied_weights=True)
plt.tight_layout()
plt.show()

valid = valid[:16*16]
plot_images(valid, 121, 'Sample data')
plot_images(e.network.predict(valid), 122, 'Reconstructed data')
plt.tight_layout()
plt.show()

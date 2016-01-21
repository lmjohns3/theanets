#!/usr/bin/env python

import matplotlib.pyplot as plt
import theanets

from utils import load_mnist, plot_layers, plot_images


train, valid, _ = load_mnist()

net = theanets.Autoencoder(
    layers=(784, 256, 100, 64, ('tied', 100), ('tied', 256), ('tied', 784)),
)
net.train(train, valid,
          algo='layerwise',
          patience=1,
          min_improvement=0.05,
          train_batches=100)
net.train(train, valid, min_improvment=0.01, train_batches=100)

plot_layers([net.find(i, 'w') for i in (1, 2, 3)], tied_weights=True)
plt.tight_layout()
plt.show()

valid = valid[0][:100]
plot_images(valid, 121, 'Sample data')
plot_images(net.predict(valid), 122, 'Reconstructed data')
plt.tight_layout()
plt.show()

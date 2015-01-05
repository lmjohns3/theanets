#!/usr/bin/env python

import climate
import matplotlib.pyplot as plt
import numpy as np
import theanets

from utils import load_mnist, plot_layers, plot_images

logging = climate.get_logger('mnist-rica')

climate.enable_default_logging()


class RICA(theanets.Autoencoder):
    def J(self, weight_inverse=0, **kwargs):
        cost = super(RICA, self).J(**kwargs)
        if weight_inverse > 0:
            cost += sum((weight_inverse / (w * w).sum(axis=0)).sum()
                        for l in self.layers for w in l.weights)
        return cost


train, valid, _ = load_mnist()

# mean-center the digits and compute a pca whitening transform.

train -= 0.5
valid -= 0.5

logging.info('computing whitening transform')
vals, vecs = np.linalg.eigh(np.dot(train.T, train) / len(train))
vals = vals[::-1]
vecs = vecs[:, ::-1]

K = 197  # this retains 99% of the variance in the digit data.
vals = np.sqrt(vals[:K])
vecs = vecs[:, :K]

def whiten(x):
    return np.dot(x, np.dot(vecs, np.diag(1. / vals)))

def color(z):
    return np.dot(z, np.dot(np.diag(vals), vecs.T))

# now train our model on the whitened dataset.

N = 16

e = theanets.Experiment(
    RICA,
    layers=(K, N * N, K),
    activation='linear',
    hidden_l1=0.2,
    no_learn_biases=True,
    tied_weights=True,
    train_batches=100,
    weight_inverse=0.01,
)
e.train(whiten(train), whiten(valid))

# color the network weights so they are viewable as digits.
plot_layers(
    [color(e.network.get_weights('hid1').T).T],
    tied_weights=True)
plt.tight_layout()
plt.show()

plot_images(valid[:N*N], 121, 'Sample data')
plot_images(
    color(e.network.predict(whiten(valid[:N*N]))),
    122, 'Reconstructed data')
plt.tight_layout()
plt.show()

#!/usr/bin/env python

import climate
import matplotlib.pyplot as plt
import numpy as np
import theanets

from utils import load_mnist, plot_layers, plot_images

logging = climate.get_logger('mnist-rica')

climate.enable_default_logging()


class RICA(theanets.Autoencoder):
    def loss(self, weight_inverse=0, **kwargs):
        loss = super(RICA, self).loss(**kwargs)
        if weight_inverse > 0:
            loss += sum((weight_inverse / (w * w).sum(axis=0)).sum()
                        for l in self.layers for w in l.params
                        if w.ndim > 1)
        return loss


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

N = 20

e = theanets.Experiment(
    RICA, layers=(K, (N * N, 'linear'), (K, 'tied')))

e.train(whiten(train),
        whiten(valid),
        hidden_l1=0.001,
        weight_inverse=1e-6,
        train_batches=300,
        monitors={'hid1:out': (-0.9, -0.1, 0.1, 0.9)})

# color the network weights so they are viewable as digits.
plot_layers([color(e.network.find('hid1', 'w').get_value().T).T], tied_weights=True)
plt.tight_layout()
plt.show()

plot_images(valid[:N*N], 121, 'Sample data')
plot_images(color(e.network.predict(whiten(valid[:N*N]))), 122, 'Reconstructed data')
plt.tight_layout()
plt.show()

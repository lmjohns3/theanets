#!/usr/bin/env python

import climate
import matplotlib.pyplot as plt
import numpy as np
import theanets

from utils import load_cifar, plot_layers, plot_images

logging = climate.get_logger('cifar')

g = climate.add_arg_group('CIFAR Example')
g.add_argument('--features', type=int, default=0, metavar='N',
               help='train a model using N^2 hidden-layer features')


K = 655  # this retains 99% of the variance in the cifar images.

def pca(dataset):
    mean = dataset[:3000].mean(axis=0)

    logging.info('computing whitening transform')
    x = dataset[:3000] - mean
    vals, vecs = np.linalg.eigh(np.dot(x.T, x) / len(x))
    vals = vals[::-1]
    vecs = vecs[:, ::-1]

    vals = np.sqrt(vals[:K])
    vecs = vecs[:, :K]

    def whiten(x):
        return np.dot(x, np.dot(vecs, np.diag(1. / vals)))

    def color(z):
        return np.dot(z, np.dot(np.diag(vals), vecs.T))

    return whiten, color


def main(args):
    train, valid, _ = load_cifar()

    whiten, color = pca(train)

    feat = args.features or int(np.sqrt(4 * K))
    e = theanets.Experiment(
        theanets.Autoencoder,
        layers=(K, feat ** 2, K),
        hidden_l1=1,
    )

    e.train(whiten(train), whiten(valid))

    plot_layers([
        color(e.network.get_weights(1).T).T,
        color(e.network.get_weights('out'))], channels=3)
    plt.tight_layout()
    plt.show()

    valid = whiten(valid[:100])
    plot_images(color(valid), 121, 'Sample data', channels=3)
    plot_images(color(e.network.predict(valid)), 122,
                'Reconstructed data', channels=3)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    climate.call(main)

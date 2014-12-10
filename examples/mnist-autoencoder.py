#!/usr/bin/env python

import climate
import matplotlib.pyplot as plt
import theanets

from utils import load_mnist, plot_layers, plot_images

g = climate.add_arg_group('MNIST Example')
g.add_argument('--features', type=int, default=8, metavar='N',
               help='train a model using N^2 hidden-layer features')


def main(args):
    train, valid, _ = load_mnist()

    e = theanets.Experiment(
        theanets.Autoencoder,
        layers=(784, args.features ** 2, 784))

    e.train(train, valid)

    plot_layers(e.network.weights)
    plt.tight_layout()
    plt.show()

    v = valid[:100]
    plot_images(v, 121, 'Sample data')
    plot_images(e.network.predict(v), 122, 'Reconstructed data')
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    climate.call(main)

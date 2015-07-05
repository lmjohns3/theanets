#!/usr/bin/env python

'''Single-layer autoencoder example using MNIST digit data.

This example shows one way to train a single-layer autoencoder model using the
handwritten MNIST digits.

This example also shows the use of climate command-line arguments.
'''

import climate
import matplotlib.pyplot as plt
import theanets

from utils import load_mnist, plot_layers, plot_images

g = climate.add_group('MNIST Example')
g.add_argument('--features', type=int, default=8, metavar='N',
               help='train a model using N^2 hidden-layer features')


def main(args):
    # load up the MNIST digit dataset.
    train, valid, _ = load_mnist()

    e = theanets.Experiment(
        theanets.Autoencoder,
        layers=(784, args.features ** 2, 784))

    e.train(train, valid,
            input_noise=0.1,
            weight_l2=0.0001,
            algorithm='rmsprop',
            momentum=0.9,
            min_improvement=0.1)

    plot_layers([e.network.find('hid1', 'w'), e.network.find('out', 'w')])
    plt.tight_layout()
    plt.show()

    v = valid[:100]
    plot_images(v, 121, 'Sample data')
    plot_images(e.network.predict(v), 122, 'Reconstructed data')
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    climate.call(main)

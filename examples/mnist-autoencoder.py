#!/usr/bin/env python

'''Single-layer autoencoder example using MNIST digit data.

This example shows one way to train a single-layer autoencoder model using the
handwritten MNIST digits.

This example also shows the use of command-line arguments.
'''

import click
import matplotlib.pyplot as plt
import theanets

from utils import load_mnist, plot_layers, plot_images

@click.command()
@click.option('--features', default=16, type=int, metavar='N',
              help='Train a model with NxN hidden features.')
def main(features):
    # load up the MNIST digit dataset.
    train, valid, _ = load_mnist()

    net = theanets.Autoencoder([784, features ** 2, 784])
    net.train(train, valid,
              train_batches=100,
              input_noise=0.1,
              weight_l2=0.0001,
              algo='rmsprop',
              momentum=0.9,
              min_improvement=0.1)

    plot_layers([net.find('hid1', 'w'), net.find('out', 'w')])
    plt.tight_layout()
    plt.show()

    v = valid[0][:100]
    plot_images(v, 121, 'Sample data')
    plot_images(net.predict(v), 122, 'Reconstructed data')
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()

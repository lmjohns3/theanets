#!/usr/bin/env python

import climate
import matplotlib.pyplot as plt
import theanets

from utils import load_cifar, plot_layers, plot_images

g = climate.add_arg_group('CIFAR Example')
g.add_argument('--features', type=int, default=32, metavar='N',
               help='train a model using N^2 hidden-layer features')


def main(args):
    train, valid, _ = load_cifar()

    e = theanets.Experiment(
        theanets.Autoencoder,
        layers=(3072, args.features ** 2, 3072))

    e.train(train, valid)

    plot_layers([e.network.get_weights(1), e.network.get_weights('out')], channels=3)
    plt.tight_layout()
    plt.show()

    valid = valid[:100]
    plot_images(valid, 121, 'Sample data', channels=3)
    plot_images(e.network.predict(valid), 122, 'Reconstructed data', channels=3)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    climate.call(main)

import climate
import pickle
import gzip
import numpy as np
import os
import tempfile

logging = climate.get_logger(__name__)

climate.enable_default_logging()

try:
    import matplotlib.pyplot as plt
except ImportError:
    logging.critical('please install matplotlib to run the examples!')
    raise

try:
    import skdata.mnist
except ImportError:
    logging.critical('please install skdata to run the examples!')
    raise


def load_mnist(labels=False):
    '''Load the MNIST digits dataset.'''
    mnist = skdata.mnist.dataset.MNIST()
    mnist.meta  # trigger download if needed.
    def arr(n, dtype):
        arr = mnist.arrays[n]
        return arr.reshape((len(arr), -1)).astype(dtype)
    train_images = arr('train_images', 'f') / 255.
    train_labels = arr('train_labels', np.uint8)
    test_images = arr('test_images', 'f') / 255.
    test_labels = arr('test_labels', np.uint8)
    if labels:
        return ((train_images[:50000], train_labels[:50000, 0]),
                (train_images[50000:], train_labels[50000:, 0]),
                (test_images, test_labels[:, 0]))
    return train_images[:50000], train_images[50000:], test_images


def plot_images(imgs, loc, title=None):
    '''Plot an array of images.

    We assume that we are given a matrix of data whose shape is (n*n, s*s) --
    that is, there are n^2 images along the first axis of the array, and each
    image is a square measuring s pixels on a side. Each row of the input will
    be plotted as a sub-region within a single image array containing an n x n
    grid of images.
    '''
    n = int(np.sqrt(len(imgs)))
    assert n * n == len(imgs), 'images array must contain a square number of rows!'
    s = int(np.sqrt(len(imgs[0])))
    assert s * s == len(imgs[0]), 'images must be square!'

    img = np.zeros((s * n, s * n), dtype=imgs[0].dtype)
    for i, pix in enumerate(imgs):
        r, c = divmod(i, n)
        img[r * s:(r+1) * s, c * s:(c+1) * s] = pix.reshape((s, s))

    ax = plt.gcf().add_subplot(loc)
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    ax.set_frame_on(False)
    ax.imshow(img, cmap=plt.cm.gray)
    if title:
        ax.set_title(title)


def plot_layers(weights, tied_weights=False):
    '''Create a plot of weights, visualized as "bottom-level" pixel arrays.'''
    if hasattr(weights[0], 'get_value'):
        weights = [w.get_value() for w in weights]
    k = min(len(weights), 9)
    imgs = np.eye(weights[0].shape[0])
    for i, weight in enumerate(weights[:-1]):
        imgs = np.dot(weight.T, imgs)
        plot_images(imgs, 100 + 10 * k + i + 1, 'Layer {}'.format(i+1))
    weight = weights[-1]
    if int(np.sqrt(weight.shape[1])) ** 2 != weight.shape[1]:
        return
    if tied_weights:
        imgs = np.dot(weight.T, imgs)
        plot_images(imgs, 100 + 10 * k + k, 'Layer {}'.format(k))
    else:
        plot_images(weight, 100 + 10 * k + k, 'Decoding weights')

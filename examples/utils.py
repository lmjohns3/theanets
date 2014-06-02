import climate
import pickle
import gzip
import matplotlib.pyplot as plt
import numpy as np
import os
import tempfile

KW = {}
try:
    import urllib.request
    KW['encoding'] = 'latin1'
except: # Python 2.x
    import urllib

logging = climate.get_logger(__name__)

climate.enable_default_logging()


def load_mnist(
        labels=False,
        url='http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz',
        local=os.path.join(tempfile.gettempdir(), 'mnist.pkl.gz')):
    '''Load the MNIST digits dataset.'''
    if not os.path.isfile(local):
        logging.info('downloading mnist digit dataset from %s' % url)
        try:
            urllib.request.urlretrieve(url, local)
        except: # Python 2.x
            urllib.urlretrieve(url, local)
        logging.info('saved mnist digits to %s' % local)
    dig = [(x, y.astype('int32')) for x, y in pickle.load(gzip.open(local), **KW)]
    if not labels:
        dig = [x[0] for x in dig]
    return dig


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

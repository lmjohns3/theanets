# -*- coding: utf-8 -*-

'''This module contains optimization methods for neural networks.

Many optimization methods are general-purpose optimization routines that happen
to be pretty good for training neural networks; these are provided by
``downhill``. The other methods here --- :class:`SampleTrainer`,
:class:`SupervisedPretrainer`, and :class:`UnsupervisedPretrainer` --- are more
specific to neural networks, often taking advantage of the layered structure of
many common network architectures.
'''

import climate
import downhill
import itertools
import numpy as np

from . import layers

logging = climate.get_logger(__name__)


class DownhillTrainer(object):
    '''Wrapper for using trainers from ``downhill``.
    '''

    def __init__(self, algo, network):
        self.algo = algo
        self.network = network

    def itertrain(self, train, valid=None, **kwargs):
        '''Train a model using a training and validation set.

        This method yields a series of monitor values to the caller. After every
        iteration, a pair of monitor dictionaries is generated: one evaluated on
        the training dataset, and another evaluated on the validation dataset.
        The validation monitors might not be updated during every training
        iteration; in this case, the most recent validation monitors will be
        yielded along with the training monitors.

        Parameters
        ----------
        train : :class:`Dataset <theanets.dataset.Dataset>`
            A set of training data for computing updates to model parameters.
        valid : :class:`Dataset <theanets.dataset.Dataset>`
            A set of validation data for computing monitor values and
            determining when the loss has stopped improving.

        Yields
        ------
        training : dict
            A dictionary mapping monitor names to values, evaluated on the
            training dataset.
        validation : dict
            A dictionary containing monitor values evaluated on the validation
            dataset.
        '''
        for monitors in downhill.build(
                algo=self.algo,
                loss=self.network.loss(**kwargs),
                updates=self.network.updates(**kwargs),
                monitors=self.network.monitors(**kwargs),
                inputs=self.network.variables,
                params=self.network.params,
                monitor_gradients=kwargs.get('monitor_gradients', False),
        ).iterate(train, valid=valid, **kwargs):
            yield monitors


class SampleTrainer(object):
    '''This trainer replaces network weights with samples from the input.'''

    @staticmethod
    def reservoir(xs, n, rng):
        '''Select a random sample of n items from xs.'''
        pool = []
        for i, x in enumerate(xs):
            if len(pool) < n:
                pool.append(x / np.linalg.norm(x))
                continue
            j = rng.randint(i + 1)
            if j < n:
                pool[j] = x / np.linalg.norm(x)
        # if the pool still has fewer than n items, pad with distorted random
        # duplicates from the source data.
        L = len(pool)
        S = np.std(pool, axis=0)
        while len(pool) < n:
            x = pool[rng.randint(L)]
            pool.append(x + S * rng.randn(*x.shape))
        return np.array(pool, dtype=pool[0].dtype)

    def __init__(self, network):
        self.network = network

    def itertrain(self, train, valid=None, **kwargs):
        '''Train a model using a training and validation set.

        This method yields a series of monitor values to the caller. After every
        iteration, a pair of monitor dictionaries is generated: one evaluated on
        the training dataset, and another evaluated on the validation dataset.
        The validation monitors might not be updated during every training
        iteration; in this case, the most recent validation monitors will be
        yielded along with the training monitors.

        Parameters
        ----------
        train : :class:`Dataset <theanets.dataset.Dataset>`
            A set of training data for computing updates to model parameters.
        valid : :class:`Dataset <theanets.dataset.Dataset>`
            A set of validation data for computing monitor values and
            determining when the loss has stopped improving.

        Yields
        ------
        training : dict
            A dictionary mapping monitor names to values, evaluated on the
            training dataset.
        validation : dict
            A dictionary containing monitor values evaluated on the validation
            dataset.
        '''
        ifci = itertools.chain.from_iterable

        def first(x):
            return x[0] if isinstance(x, (tuple, list)) else x

        def last(x):
            return x[-1] if isinstance(x, (tuple, list)) else x

        odim = idim = None
        for t in train:
            idim = first(t).shape[-1]
            odim = last(t).shape[-1]

        rng = kwargs.get('rng')
        if rng is None or isinstance(rng, int):
            rng = np.random.RandomState(rng)

        # set output (decoding) weights on the network.
        samples = ifci(last(t) for t in train)
        for param in self.network.layers[-1].params:
            shape = param.get_value(borrow=True).shape
            if len(shape) == 2 and shape[1] == odim:
                arr = np.vstack(SampleTrainer.reservoir(samples, shape[0], rng))
                logging.info('setting %s: %s', param.name, shape)
                param.set_value(arr / np.sqrt((arr * arr).sum(axis=1))[:, None])

        # set input (encoding) weights on the network.
        samples = ifci(first(t) for t in train)
        for layer in self.network.layers:
            for param in layer.params:
                shape = param.get_value(borrow=True).shape
                if len(shape) == 2 and shape[0] == idim:
                    arr = np.vstack(SampleTrainer.reservoir(samples, shape[1], rng)).T
                    logging.info('setting %s: %s', param.name, shape)
                    param.set_value(arr / np.sqrt((arr * arr).sum(axis=0)))
                    samples = ifci(self.network.feed_forward(
                        first(t))[i-1] for t in train)

        yield dict(loss=0), dict(loss=0)


class SupervisedPretrainer(object):
    '''This trainer adapts parameters using a supervised pretraining approach.

    In this variant, we create "taps" at increasing depths into the original
    network weights, training only those weights that are below the tap. So, for
    a hypothetical binary classifier network with layers [3, 4, 5, 6, 2], we
    would first insert a tap after the first hidden layer (effectively a binary
    classifier in a [3, 4, (2)] configuration, where (2) indicates that the
    corresponding layer is the tap, not present in the original) and train just
    that network. Then we insert a tap at the next layer (effectively training a
    [3, 4, 5, (2)] classifier, re-using the trained weights for the 3 x 4
    layer), and so forth. When we get to training the last layer, i.e., [3, 4,
    5, 6, 2], then we just train all of the layers in the original network.

    For autoencoder networks with tied weights, consider an example with layers
    [3, 4, 5, 6, 5', 4', 3'], where the prime indicates that the layer is tied.
    In cases like this, we train the "outermost" pair of layers first, then add
    then next pair of layers inward, etc. The training for our example would
    start with [3, 4, 3'], then proceed to [3, 4, 5, 4', 3'], and then finish by
    training all the layers in the original network.

    By using layers from the original network whenever possible, we preserve all
    of the relevant settings of noise, dropouts, loss function and the like, in
    addition to removing the need for copying trained weights around between
    different :class:`Network <theanets.graph.Network>` instances.

    References
    ----------

    .. [Ben06] Y. Bengio, P. Lamblin, D. Popovici, & H. Larochelle. (NIPS 2006)
       "Greedy Layer-Wise Training of Deep Networks"
       http://machinelearning.wustl.edu/mlpapers/paper_files/NIPS2006_739.pdf

       The Appendix also contains pseudocode for the approaches:
       http://www.iro.umontreal.ca/~pift6266/A06/refs/appendix_dbn_supervised.pdf
    '''

    def __init__(self, algo, network):
        self.algo = algo
        self.network = network

    def itertrain(self, train, valid=None, **kwargs):
        '''Train a model using a training and validation set.

        This method yields a series of monitor values to the caller. After every
        iteration, a pair of monitor dictionaries is generated: one evaluated on
        the training dataset, and another evaluated on the validation dataset.
        The validation monitors might not be updated during every training
        iteration; in this case, the most recent validation monitors will be
        yielded along with the training monitors.

        Parameters
        ----------
        train : :class:`Dataset <theanets.dataset.Dataset>`
            A set of training data for computing updates to model parameters.
        valid : :class:`Dataset <theanets.dataset.Dataset>`
            A set of validation data for computing monitor values and
            determining when the loss has stopped improving.

        Yields
        ------
        training : dict
            A dictionary mapping monitor names to values, evaluated on the
            training dataset.
        validation : dict
            A dictionary containing monitor values evaluated on the validation
            dataset.
        '''
        net = self.network
        original = list(net.layers)
        output_name = original[-1].output_name
        tied = any(isinstance(l, layers.Tied) for l in original)
        L = 1 + len(original) // 2 if tied else len(original) - 1
        for i in range(1, L):
            tail = []
            if i == L - 1:
                net.layers = original
            elif tied:
                net.layers = original[:i+1]
                for j in range(i):
                    prev = tail[-1] if tail else net.layers[-1]
                    tail.append(layers.Layer.build(
                        'tied', partner=original[i-j].name, inputs=prev.name))
                net.layers = original[:i+1] + tail
            else:
                tail.append(layers.Layer.build(
                    'feedforward',
                    name='lwout',
                    inputs=original[i].output_name,
                    size=original[-1].size,
                    activation=original[-1].kwargs['activation']))
                net.layers = original[:i+1] + tail
            logging.info('layerwise: training %s',
                         ' -> '.join(l.name for l in net.layers))
            [l.resolve(net.layers) for l in net.layers]
            [l.setup() for l in tail]
            [l.log() for l in net.layers]
            net.losses[0].output_name = net.layers[-1].output_name
            trainer = DownhillTrainer(self.algo, net)
            for monitors in trainer.itertrain(train, valid, **kwargs):
                yield monitors
        net.layers = original
        net.losses[0].output_name = output_name


class UnsupervisedPretrainer(object):
    '''Train a classification model using an unsupervised pre-training step.

    This trainer is a bit of glue code that creates a "shadow" autoencoder based
    on a current network model, trains the autoencoder, and then transfers the
    trained weights back to the original model.

    This code is intended mostly as a proof-of-concept to demonstrate how shadow
    networks can be created, and how trainers can call other trainers for lots
    of different types of training regimens.
    '''

    def __init__(self, algo, network):
        self.algo = algo
        self.network = network

    def itertrain(self, train, valid=None, **kwargs):
        '''Train a model using a training and validation set.

        This method yields a series of monitor values to the caller. After every
        iteration, a pair of monitor dictionaries is generated: one evaluated on
        the training dataset, and another evaluated on the validation dataset.
        The validation monitors might not be updated during every training
        iteration; in this case, the most recent validation monitors will be
        yielded along with the training monitors.

        Parameters
        ----------
        train : :class:`Dataset <theanets.dataset.Dataset>`
            A set of training data for computing updates to model parameters.
        valid : :class:`Dataset <theanets.dataset.Dataset>`
            A set of validation data for computing monitor values and
            determining when the loss has stopped improving.

        Yields
        ------
        training : dict
            A dictionary mapping monitor names to values, evaluated on the
            training dataset.
        validation : dict
            A dictionary containing monitor values evaluated on the validation
            dataset.
        '''
        from . import feedforward

        # construct a "shadow" of the input network, using the original
        # network's encoding layers, with tied weights in an autoencoder
        # configuration.
        layers_ = list(l.to_spec() for l in self.network.layers[:-1])
        for i, l in enumerate(layers_[::-1][:-2]):
            layers_.append(dict(
                form='tied', partner=l['name'], activation=l['activation']))
        layers_.append(dict(
            form='tied', partner=layers_[1]['name'], activation='linear'))

        logging.info('creating shadow network')
        ae = feedforward.Autoencoder(layers=layers_)

        # train the autoencoder using the supervised layerwise pretrainer.
        pre = SupervisedPretrainer(self.algo, ae)
        for monitors in pre.itertrain(train, valid, **kwargs):
            yield monitors

        # copy trained parameter values back to our original network.
        for param in ae.params:
            if not param.name.startswith('tied'):
                l, p = param.name.split('.')
                logging.info('copying pretrained parameter %s', param.name)
                self.network.find(l, p).set_value(param.get_value())

        logging.info('completed unsupervised pretraining')

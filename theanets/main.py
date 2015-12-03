'''This module contains some glue code encapsulating a "main" process.

The code here is aimed at wrapping the most common tasks involved in creating
and, especially, training a neural network model.
'''

import climate
import datetime
import downhill
import os
import warnings

from . import graph
from . import trainer

logging = climate.get_logger(__name__)


class Experiment:
    '''This class encapsulates tasks for training and evaluating a network.

    Parameters
    ----------
    model : :class:`Network <graph.Network>` or str
        A specification for obtaining a model. If a string is given, it is
        assumed to name a file containing a pickled model; this file will be
        loaded and used. If a network instance is provided, it will be used
        as the model. If a callable (such as a subclass) is provided, it
        will be invoked using the provided keyword arguments to create a
        network instance.
    '''

    def __init__(self, network, *args, **kwargs):
        if isinstance(network, str) and os.path.isfile(network):
            self.load(network)
        elif isinstance(network, graph.Network):
            self.network = network
        else:
            assert network is not graph.Network, \
                'use a concrete theanets.Network subclass ' \
                'like theanets.{Autoencoder,Regressor,...}'
            self.network = network(*args, **kwargs)

    def create_trainer(self, train, algo='rmsprop'):
        '''Create a trainer.

        Additional keyword arguments are passed directly to the trainer.

        Parameters
        ----------
        train : str
            A string describing a trainer to use.
        algo : str
            A string describing an optimization algorithm.

        Returns
        -------
        trainer : :class:`Trainer <trainer.Trainer>`
            A trainer instance to alter the parameters of our network.
        '''
        train = train.lower()
        if train == 'sample':
            return trainer.SampleTrainer(self.network)
        if train.startswith('layer') or train.startswith('sup'):
            return trainer.SupervisedPretrainer(algo, self.network)
        if train.startswith('pre') or train.startswith('unsup'):
            return trainer.UnsupervisedPretrainer(algo, self.network)
        return trainer.DownhillTrainer(train, self.network)

    def create_dataset(self, data, **kwargs):
        '''Create a dataset for this experiment.

        Parameters
        ----------
        data : sequence of ndarray or callable
            The values that you provide for data will be encapsulated inside a
            :class:`Dataset <downhill.Dataset>` instance; see that class for
            documentation on the types of things it needs. In particular, you
            can currently pass in either a list/array/etc. of data, or a
            callable that generates data dynamically.

        Returns
        -------
        data : :class:`Dataset <downhill.Dataset>`
            A dataset capable of providing mini-batches of data to a training
            algorithm.
        '''
        default_axis = 0
        if not callable(data) and not callable(data[0]) and len(data[0].shape) == 3:
            default_axis = 1
        name = kwargs.get('name', 'dataset')
        b, i, s = 'batch_size', 'iteration_size', '{}_batches'.format(name)
        return downhill.Dataset(
            data,
            name=name,
            batch_size=kwargs.get(b, 32),
            iteration_size=kwargs.get(i, kwargs.get(s)),
            axis=kwargs.get('axis', default_axis),
            rng=kwargs.get('rng', self.network._rng),
        )

    def train(self, *args, **kwargs):
        '''Train the network until the trainer converges.

        All arguments are passed to :func:`itertrain`.

        Returns
        -------
        training : dict
            A dictionary of monitor values computed using the training dataset,
            at the conclusion of training. This dictionary will at least contain
            a 'loss' key that indicates the value of the loss function. Other
            keys may be available depending on the trainer being used.
        validation : dict
            A dictionary of monitor values computed using the validation
            dataset, at the conclusion of training.
        '''
        monitors = None
        for monitors in self.itertrain(*args, **kwargs):
            pass
        return monitors

    def itertrain(self, train, valid=None, algorithm='rmsprop', **kwargs):
        '''Train our network, one batch at a time.

        This method yields a series of ``(train, valid)`` monitor pairs. The
        ``train`` value is a dictionary mapping names to monitor values
        evaluated on the training dataset. The ``valid`` value is also a
        dictionary mapping names to values, but these values are evaluated on
        the validation dataset.

        Because validation might not occur every training iteration, the
        validation monitors might be repeated for multiple training iterations.
        It is probably most helpful to think of the validation monitors as being
        the "most recent" values that have been computed.

        After training completes, the network attribute of this class will
        contain the trained network parameters.

        Parameters
        ----------
        train : sequence of ndarray or :class:`downhill.Dataset`
            A dataset to use when training the network. If this is a
            ``downhill.Dataset`` instance, it will be used directly as the
            training datset. If it is another type, like a numpy array, it will
            be converted to a ``downhill.Dataset`` and then used as the training
            set.
        valid : sequence of ndarray or :class:`downhill.Dataset`, optional
            If this is provided, it will be used as a validation dataset. If not
            provided, the training set will be used for validation. (This is not
            recommended!)
        algorithm : str or list of str, optional
            One or more optimization algorithms to use for training our network.
            If not provided, RMSProp will be used.

        Yields
        ------
        training : dict
            A dictionary of monitor values computed using the training dataset,
            at the conclusion of training. This dictionary will at least contain
            a 'loss' key that indicates the value of the loss function. Other
            keys may be available depending on the trainer being used.
        validation : dict
            A dictionary of monitor values computed using the validation
            dataset, at the conclusion of training.
        '''
        # set up datasets
        if valid is None:
            valid = train
        if not isinstance(valid, downhill.Dataset):
            valid = self.create_dataset(valid, name='valid', **kwargs)
        if not isinstance(train, downhill.Dataset):
            train = self.create_dataset(train, name='train', **kwargs)

        # set up training algorithm(s)
        if 'optimize' in kwargs:
            warnings.warn(
                'please use the "algorithm" keyword arg instead of "optimize"',
                DeprecationWarning)
            algorithm = kwargs.pop('optimize')
        if isinstance(algorithm, str):
            algorithm = algorithm.split()

        # set up auto-saving if enabled
        progress = kwargs.get('save_progress')
        timeout = kwargs.get('save_every', 0)
        if timeout < 0:  # timeout < 0 is in minutes instead of iterations.
            timeout *= 60

        # loop over trainers, saving every N minutes/iterations if enabled
        for algo in algorithm:
            if not callable(getattr(algo, 'itertrain', None)):
                algo = self.create_trainer(algo)
            start = datetime.datetime.now()
            for i, monitors in enumerate(algo.itertrain(train, valid, **kwargs)):
                yield monitors
                now = datetime.datetime.now()
                elapsed = (now - start).total_seconds()
                if i and progress and (
                        (timeout < 0 and elapsed > -timeout) or
                        (timeout > 0 and i % int(timeout) == 0)):
                    self.save(progress)
                    start = now

    def save(self, path):
        '''Save the current network to a pickle file on disk.

        Parameters
        ----------
        path : str
            Location of the file to save the network.
        '''
        self.network.save(path)

    def load(self, path):
        '''Load a saved network from a pickle file on disk.

        This method sets the ``network`` attribute of the experiment to the
        loaded network model.

        Parameters
        ----------
        filename : str
            Load the keyword arguments and parameters of a network from a pickle
            file at the named path. If this name ends in ".gz" then the input
            will automatically be gunzipped; otherwise the input will be treated
            as a "raw" pickle.

        Returns
        -------
        network : :class:`Network <graph.Network>`
            A newly-constructed network, with topology and parameters loaded
            from the given pickle file.
        '''
        self.network = graph.Network.load(path)
        return self.network

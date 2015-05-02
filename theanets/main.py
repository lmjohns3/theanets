'''This module contains some glue code encapsulating a "main" process.

The code here is aimed at wrapping the most common tasks involved in creating
and, especially, training a neural network model.
'''

import climate
import datetime
import os

from . import dataset
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

    def create_trainer(self, factory, *args, **kwargs):
        '''Create a trainer.

        Additional positional and keyword arguments are passed directly to the
        trainer factory.

        Parameters
        ----------
        factory : str or callable
            A callable that creates a trainer, or a string that maps to a
            trainer constructor.

        Returns
        -------
        trainer : :class:`Trainer <trainer.Trainer>`
            A trainer instance to alter the parameters of our network.
        '''
        args = (self.network, ) + args
        if isinstance(factory, str):
            if factory.lower() in trainer.Scipy.METHODS:
                args = (self.network, factory)
                factory = trainer.Scipy
            elif factory.lower().startswith('layer'):
                if len(args) == 1:
                    # use RmsProp trainer by default for individual layers
                    args += (trainer.RmsProp, )
                factory = trainer.SupervisedPretrainer
            elif factory.lower().startswith('pre'):
                if len(args) == 1:
                    # use RmsProp trainer by default for pretrainer
                    args += (trainer.RmsProp, )
                factory = trainer.UnsupervisedPretrainer
            else:
                factory = dict(
                    adadelta=trainer.ADADELTA,
                    esgd=trainer.ESGD,
                    hf=trainer.HF,
                    nag=trainer.NAG,
                    rmsprop=trainer.RmsProp,
                    rprop=trainer.Rprop,
                    sample=trainer.Sample,
                    sgd=trainer.SGD,
                )[factory.lower()]
        logging.info('creating trainer %s', factory)
        for k in sorted(kwargs):
            logging.info('--%s = %s', k, kwargs[k])
        return factory(*args, **kwargs)

    def create_dataset(self, data, **kwargs):
        '''Create a dataset for this experiment.

        Parameters
        ----------
        data : ndarray, (ndarray, ndarray), or callable
            The values that you provide for data will be encapsulated inside a
            :class:`Dataset <dataset.Dataset>` instance; see that class for
            documentation on the types of things it needs. In particular, you
            can currently pass in either a list/array/etc. of data, or a
            callable that generates data dynamically.

        Returns
        -------
        data : :class:`Dataset <dataset.Dataset>`
            A dataset capable of providing mini-batches of data to a training
            algorithm.
        '''
        samples, labels, weights = data, None, None
        if isinstance(data, (tuple, list)):
            if len(data) > 0:
                samples = data[0]
            if len(data) > 1:
                labels = data[1]
            if len(data) > 2:
                weights = data[2]
        name = kwargs.get('name', 'dataset')
        b, i, s = 'batch_size', 'iteration_size', '{}_batches'.format(name)
        return dataset.Dataset(
            samples, labels=labels, weights=weights, name=name,
            batch_size=kwargs.get(b, 32),
            iteration_size=kwargs.get(i, kwargs.get(s)),
            axis=kwargs.get('axis'))

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

    def itertrain(self, train_set=None, valid_set=None, optimize=None, **kwargs):
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
        train_set : any
            A dataset to use when training the network. If this is a `Dataset`
            instance, it will be used directly as the training datset. If it is
            another type, like a numpy array, it will be converted to a
            `Dataset` and then used as the training set.
        valid_set : any, optional
            If this is provided, it will be used as a validation dataset. If not
            provided, the training set will be used for validation. (This is not
            recommended!)
        optimize : any, optional
            One or more optimization algorithms to use for training our network.
            If this is not provided, then optimizers will be created based on
            command-line arguments. If neither are provided, NAG will be used.

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
        # set up datasets
        if valid_set is None:
            valid_set = train_set
        if not isinstance(valid_set, dataset.Dataset):
            valid_set = self.create_dataset(valid_set, name='valid', **kwargs)
        if not isinstance(train_set, dataset.Dataset):
            train_set = self.create_dataset(train_set, name='train', **kwargs)
        sets = dict(train_set=train_set, valid_set=valid_set, cg_set=train_set)

        # set up training algorithm(s)
        optimize = optimize or 'rmsprop'
        if isinstance(optimize, str):
            optimize = optimize.split()

        # set up auto-saving if enabled
        progress = kwargs.get('save_progress')
        timeout = kwargs.get('save_every', 0)
        if timeout < 0:  # timeout < 0 is in minutes instead of iterations.
            timeout *= 60

        # loop over trainers, saving every N minutes/iterations if enabled
        for opt in optimize:
            if not callable(getattr(opt, 'train', None)):
                opt = self.create_trainer(opt, **kwargs)
            start = datetime.datetime.now()
            for i, monitors in enumerate(opt.itertrain(**sets)):
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
        logging.info('%s: saving model', path)
        self.network.save(path)

    def load(self, path, **kwargs):
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
        logging.info('%s: loading model', path)
        self.network = graph.load(path, **kwargs)
        return self.network

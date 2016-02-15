'''This module contains some glue code encapsulating a "main" process.

The code here wraps the most common tasks involved in creating and, especially,
training a neural network model.
'''

import climate
import os

from . import graph
from . import util

logging = climate.get_logger(__name__)


class Experiment:
    '''This class encapsulates tasks for training and evaluating a network.

    Parameters
    ----------
    model : :class:`Network <theanets.graph.Network>` or str
        A specification for obtaining a model. If a string is given, it is
        assumed to name a file containing a pickled model; this file will be
        loaded and used. If a network instance is provided, it will be used
        as the model. If a callable (such as a subclass) is provided, it
        will be invoked using the provided keyword arguments to create a
        network instance.
    '''

    def __init__(self, network, *args, **kwargs):
        if isinstance(network, util.basestring) and os.path.isfile(network):
            self.load(network)
        elif isinstance(network, graph.Network):
            self.network = network
        else:
            assert network is not graph.Network, \
                'use a concrete theanets.Network subclass ' \
                'like theanets.{Autoencoder,Regressor,...}'
            self.network = network(*args, **kwargs)

    def train(self, *args, **kwargs):
        '''Train the network until the trainer converges.

        All arguments are passed to :func:`train
        <theanets.graph.Network.itertrain>`.

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
        return self.network.train(*args, **kwargs)

    def itertrain(self, *args, **kwargs):
        '''Train the network iteratively.

        All arguments are passed to :func:`itertrain
        <theanets.graph.Network.itertrain>`.

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
        return self.network.itertrain(*args, **kwargs)

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

# Copyright (c) 2012 Leif Johnson <leif@leifjohnson.net>
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

'''This file contains an object encapsulating a main process.'''

import climate
import theano.tensor as TT

from .dataset import SequenceDataset as Dataset
from . import feedforward
from . import trainer

logging = climate.get_logger(__name__)


def parse_args(**overrides):
    '''Parse command-line arguments, overriding with keyword arguments.

    Returns an ordered pair of the command-line argument structure, as well as a
    dictionary version of these arguments.
    '''
    args = climate.get_args().parse_args()
    for k, v in overrides.items():
        setattr(args, k, v)
    kwargs = {}
    kwargs.update(vars(args))
    logging.info('runtime arguments:')
    for k in sorted(kwargs):
        logging.info('--%s = %s', k, kwargs[k])
    return args, kwargs


class Experiment(object):
    '''This class encapsulates tasks for training and evaluating a network.'''

    def __init__(self, network_class, **overrides):
        '''Set up an experiment -- build a network and a trainer.

        The only input this constructor needs is the Python class of the network
        to build. Other configuration---for example, creating the appropriate
        trainer class---typically takes place by parsing command-line argument
        values.

        Datasets also need to be added to the experiment, either :

        - manually, by calling add_dataset(...), or
        - at runtime, by providing data to the run(train_data, valid_data)
          method.

        Datasets are typically provided as numpy arrays, but they can also be
        provided as callables, as described in the dataset module.

        Any keyword arguments provided to the constructor will be used to
        override values passed on the command line. (Typically this is used to
        provide experiment-specific default values for command line arguments
        that have no global defaults, e.g., network architecture.)
        '''
        self.trainers = []
        self.datasets = {}

        self.args, self.kwargs = parse_args(**overrides)

        kw = {}
        kw.update(self.kwargs)
        self.network = self._build_network(network_class, **kw)

        kw = {}
        kw.update(self.kwargs)
        self._build_trainers(**kw)

    def _build_network(self, network_class, **kwargs):
        '''Build a Network class instance to compute input transformations.
        '''
        assert network_class is not feedforward.Network, \
            'use a concrete theanets.Network subclass ' \
            'like theanets.{Autoencoder,Regressor,...}'
        return network_class(**kwargs)

    def _build_trainers(self, **kwargs):
        '''Build trainers from command-line arguments.
        '''
        if isinstance(self.args.optimize, str):
            self.args.optimize = self.args.optimize.strip().split()
        for factory in self.args.optimize:
            self.add_trainer(factory, **kwargs)

    def add_trainer(self, factory, *args, **kwargs):
        '''Add a new trainer to this experiment.

        Arguments
        ---------
        factory : str or callable
            A callable that creates a Trainer instance, or a string that maps to
            a Trainer constructor.

        Remaining positional and keyword arguments are passed directly to the
        trainer factory.
        '''
        args = (self.network, ) + args
        if isinstance(factory, str):
            if factory.lower() in trainer.Scipy.METHODS:
                args = (self.network, factory)
                factory = trainer.Scipy
            elif factory.lower().startswith('l'):
                if len(args) == 1:
                    # use SGD trainer by default for individual layers
                    args += (trainer.SGD, )
                factory = trainer.Layerwise
            else:
                factory = dict(
                    hf=trainer.HF,
                    sample=trainer.Sample,
                    rprop=trainer.RPROP,
                    sgd=trainer.SGD
                )[factory.lower()]
        kw = {}
        kw.update(self.kwargs)
        kw.update(kwargs)
        logging.info('adding trainer %s', factory)
        for k in sorted(kwargs):
            logging.info('--%s = %s', k, kw[k])
        self.trainers.append(factory(*args, **kw))

    def add_dataset(self, label, dataset, **kwargs):
        '''Add a dataset to this experiment.

        The provided label is used to determine the type of data in the set.
        Currently this label can be :

        - train -- for training data,
        - valid -- for validation data, typically a small slice of the training
          data, or
        - cg -- for using the HF optimizer, typically using the same underlying
          data as the training set.

        Other labels can be added, but but they are not currently used.

        The value that you provide for dataset will be encapsulated inside a
        SequenceDataset instance ; see that class for documentation on the types
        of things it needs. In particular, you can currently pass in either a
        list/array/etc. of data, or a callable that generates data dynamically.
        '''
        if 'batches' not in kwargs:
            b = getattr(self.args, '%s_batches' % label, None)
            kwargs['batches'] = b
        if 'size' not in kwargs:
            kwargs['size'] = self.args.batch_size
        kwargs['label'] = label
        if not isinstance(dataset, (tuple, list)):
            dataset = (dataset, )
        self.datasets[label] = Dataset(*dataset, **kwargs)

    def run(self, train=None, valid=None):
        '''Run this experiment by training and validating our network.
        '''
        for _ in self.train(train=train, valid=valid):
            pass

    def train(self, train=None, valid=None):
        '''Train (and validate) our network.

        Before calling this method, datasets will typically need to have been
        added to the experiment by calling add_dataset(...). However, as a
        shortcut, you can provide training and validation data as arguments to
        this method, and these arguments will be used to add datasets as needed.

        Usually the output of this method is whatever is logged to the console
        during training. After training completes, the network attribute of this
        class will contain the trained network parameters.
        '''
        if not self.trainers:
            # train using sgd if no other trainer has been added.
            self.add_trainer('sgd')
        if train is not None:
            if 'train' not in self.datasets:
                self.add_dataset('train', train)
            if 'cg' not in self.datasets:
                self.add_dataset('cg', train)
        if valid is not None and 'valid' not in self.datasets:
            self.add_dataset('valid', valid)
        for trainer in self.trainers:
            for _ in trainer.train(train_set=self.datasets['train'],
                                   valid_set=self.datasets['valid'],
                                   cg_set=self.datasets['cg']):
                yield

    def save(self, path):
        '''Save the parameters in the network to a pickle file on disk.
        '''
        self.network.save(path)

    def load(self, path):
        '''Load the parameters in the network from a pickle file on disk.
        '''
        self.network.load(path)

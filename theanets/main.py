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
import sys
import theano.tensor as TT
import warnings

from .dataset import SequenceDataset as Dataset
from . import feedforward
from . import trainer

logging = climate.get_logger(__name__)


HELP_ACTIVATION = '''\
Available Activation Functions
==============================

The names on the left show the possible values for the --hidden-activation and
--output-activation configuration parameters. They can be chained together by
combining multiple names with a + (plus); for example, tanh+relu will result in
the rectified tanh activation function g(z) = max(0, tanh(z)).

linear     g(z) = z                     plain linear

sigmoid    g(z) = 1 / (1 + e^-z)        logistic sigmoid
logistic   g(z) = 1 / (1 + e^-z)        logistic sigmoid
tanh       g(z) = tanh(z)               hyperbolic tangent

softplus   g(z) = log(1 + exp(z))       smooth approximation to relu

softmax    g(z) = e^z / sum(e^z)        categorical distribution

relu       g(z) = max(0, z)             rectified linear
trel       g(z) = max(0, min(1, z))     truncated rectified linear
trec       g(z) = z if z > 1 else 0     thresholded rectified linear
tlin       g(z) = z if |z| > 1 else 0   thresholded linear

rect:max   g(z) = min(1, z)             truncation operator
rect:min   g(z) = max(0, z)             rectification operator

norm:dc    g(z) = z - mean(z)           mean-normalization operator
norm:max   g(z) = z - max(abs(z))       max-normalization operator
norm:std   g(z) = z - std(z)            variance-normalization operator
'''

HELP_OPTIMIZE = '''\
Available Optimizers
====================

First-Order Gradient Descent
----------------------------
sgd: Stochastic Gradient Descent
  --learning-rate
  --momentum

nag: Nesterov's Accelerated Gradient
  --learning-rate
  --momentum

rprop: Resilient Backpropagation
  --learning-rate (sets initial learning rate)
  --rprop-increase, --rprop-decrease
  --rprop-min-step, --rprop-max-step

rmsprop: RMS-scaled Backpropagation
  --learning-rate
  --momentum (sets decay for exponential moving average)

bfgs, cg, dogleg, newton-cg, trust-ncg
  These use the implementations in scipy.optimize.minimize.

Second-Order Gradient Descent
-----------------------------
hf: Hessian-Free
  --cg-batches
  --initial-lambda
  --global-backtracking
  --num-updates
  --preconditioner

Miscellaneous
-------------
sample: Set model parameters to training data samples

layerwise: Greedy layerwise pre-training
  This trainer applies NAG to each layer.
'''


class Experiment:
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

        self.args, self.kwargs = climate.parse_args(**overrides)
        if 'activation' in self.kwargs:
            warnings.warn(
                'please use --hidden-activation instead of --activation',
                DeprecationWarning)
            activation = self.kwargs.pop('activation')
            if not self.kwargs.get('hidden_activation'):
                self.kwargs['hidden_activation'] = activation

        if self.kwargs.get('help_activation'):
            print(HELP_ACTIVATION)
            sys.exit(0)

        if self.kwargs.get('help_optimize'):
            print(HELP_OPTIMIZE)
            sys.exit(0)

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
        if not hasattr(self.args, 'optimize'):
            self.args.optimize = 'nag'
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
                    # use NAG trainer by default for individual layers
                    args += (trainer.NAG, )
                factory = trainer.Layerwise
            else:
                factory = dict(
                    hf=trainer.HF,
                    nag=trainer.NAG,
                    rmsprop=trainer.RmsProp,
                    rprop=trainer.Rprop,
                    sample=trainer.Sample,
                    sgd=trainer.SGD,
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
            # train using NAG if no other trainer has been added.
            self.add_trainer('nag')
        if train is not None:
            if 'train' not in self.datasets:
                self.add_dataset('train', train)
            if 'cg' not in self.datasets:
                self.add_dataset('cg', train)
        if valid is not None and 'valid' not in self.datasets:
            self.add_dataset('valid', valid)
        for trainer in self.trainers:
            for costs in trainer.train(train_set=self.datasets['train'],
                                       valid_set=self.datasets['valid'],
                                       cg_set=self.datasets['cg']):
                yield costs

    def save(self, path):
        '''Save the parameters in the network to a pickle file on disk.
        '''
        self.network.save(path)

    def load(self, path):
        '''Load the parameters in the network from a pickle file on disk.
        '''
        self.network.load(path)

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

from . import dataset
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
        values, or by a call to train(...).

        Any keyword arguments provided to the constructor will be used to
        override values passed on the command line. (Typically this is used to
        provide experiment-specific default values for command line arguments
        that have no global defaults, e.g., network architecture.)
        '''
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

        assert network_class is not feedforward.Network, \
            'use a concrete theanets.Network subclass ' \
            'like theanets.{Autoencoder,Regressor,...}'
        self.network = network_class(**self.kwargs)

    def create_trainer(self, factory, *args, **kwargs):
        '''Create a trainer.

        Parameters
        ----------
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
        logging.info('creating trainer %s', factory)
        for k in sorted(kw):
            logging.info('--%s = %s', k, kw[k])
        return factory(*args, **kw)

    def create_dataset(self, label, data, **kwargs):
        '''Add a dataset to this experiment.

        The provided label is used to determine the type of data in the set.
        Currently this label can be :

        - train -- for training data,
        - valid -- for validation data, typically a small slice of the training
          data, or
        - cg -- for using the HF optimizer, typically using the same underlying
          data as the training set.

        Other labels can be added, but but they are not currently used.

        The value that you provide for data will be encapsulated inside a
        SequenceDataset instance; see that class for documentation on the types
        of things it needs. In particular, you can currently pass in either a
        list/array/etc. of data, or a callable that generates data dynamically.
        '''
        if 'batches' not in kwargs:
            kwargs['batches'] = self.kwargs.get('%s_batches' % label, None)
        if 'size' not in kwargs:
            kwargs['size'] = self.args.batch_size
        kwargs['label'] = label
        if not isinstance(data, (tuple, list)):
            data = (data, )
        return dataset.Dataset(*data, **kwargs)

    def run(self, *args, **kwargs):
        warnings.warn(
            'please use Experiment.train() instead of Experiment.run()',
            DeprecationWarning)
        return self.train(*args, **kwargs)

    def train(self, *args, **kwargs):
        '''Train the network until the trainer converges.

        All arguments are passed to `itertrain`.
        '''
        for _ in self.itertrain(*args, **kwargs):
            pass

    def itertrain(self, train_set=None, valid_set=None, optimize=None, **kwargs):
        '''Train our network, one batch at a time.

        The output of this method is whatever is logged to the console during
        training, but the method pauses after each trainer completes a training
        iteration.

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
        sequence of dict :
            This method generates a series of dictionaries that represent the
            cost values of the model being trained. Each dictionary should have
            a "J" key providing the total cost of the model with respect to the
            training dataset. Other keys are available depending on the trainer.
        '''
        if valid_set is None:
            valid_set = train_set
        if not isinstance(valid_set, dataset.Dataset):
            valid_set = self.create_dataset('valid', valid_set)
        if not isinstance(train_set, dataset.Dataset):
            train_set = self.create_dataset('train', train_set)
        sets = dict(train_set=train_set, valid_set=valid_set, cg_set=train_set)
        if optimize is None:
            optimize = self.kwargs.get('optimize')
        if not optimize:
            optimize = 'nag'  # use nag if nothing else is defined.
        if isinstance(optimize, str):
            optimize = optimize.split()
        for opt in optimize:
            if not callable(getattr(opt, 'train', None)):
                opt = self.create_trainer(opt, **kwargs)
            for costs in opt.train(**sets):
                yield costs

    def save(self, path):
        '''Save the current network to a pickle file on disk.

        Parameters
        ----------
        path : str
            Location of the file to save the network.
        '''
        self.network.save(path)

    def load(self, path, **kwargs):
        '''Load a saved network from a pickle file on disk.

        Parameters
        ----------
        filename : str
            Load the keyword arguments and parameters of a network from a pickle
            file at the named path. If this name ends in ".gz" then the input
            will automatically be gunzipped; otherwise the input will be treated
            as a "raw" pickle.

        Returns
        -------
        Network :
            A newly-constructed network, with topology and parameters loaded
            from the given pickle file.
        '''
        self.network = feedforward.load(path, **kwargs)
        return self.network

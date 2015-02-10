'''This module contains an object encapsulating a "main" process.

The code here is aimed at wrapping the most common tasks involved in creating
and, especially, training a neural network model.
'''

import climate
import datetime
import os
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
  --momentum
  --rms-halflife

adadelta: ADADELTA
  --rms-halflife

bfgs, cg, dogleg, newton-cg, trust-ncg
  These use the implementations in scipy.optimize.minimize.

Second-Order Gradient Descent
-----------------------------
hf: Hessian-Free
  --cg-batches
  --initial-lambda
  --global-backtracking
  --preconditioner

Miscellaneous
-------------
sample: Set model parameters to training data samples

layerwise: Greedy supervised layerwise pre-training
  This trainer applies RmsProp to each layer sequentially.

pretrain: Greedy unsupervised layerwise pre-training.
  This trainer applies RmsProp to a tied-weights "shadow" autoencoder using an
  unlabeled dataset, and then transfers the learned autoencoder weights to the
  model being trained.
'''


class Experiment:
    '''This class encapsulates tasks for training and evaluating a network.'''

    def __init__(self, network_class=None, **overrides):
        '''Set up an experiment by parsing arguments and building a network.

        The only input this constructor needs is the Python class of the network
        to build. Other configuration---for example, creating the appropriate
        trainer class---typically takes place by parsing command-line argument
        values, or by a call to :func:`train`.

        Any keyword arguments provided to the constructor will be used to
        override values passed on the command line. (Typically this is used to
        provide experiment-specific default values for command line arguments
        that have no global defaults, e.g., network architecture.)
        '''
        args, _ = climate.parse_known_args(**overrides)

        self.kwargs = vars(args)

        if self.kwargs.get('activation') and 'hidden_activation' not in overrides:
            warnings.warn(
                'please use --hidden-activation instead of --activation',
                DeprecationWarning)
            self.kwargs['hidden_activation'] = self.kwargs.pop('activation')

        if self.kwargs.get('help_activation'):
            print(HELP_ACTIVATION)
            sys.exit(0)

        if self.kwargs.get('help_optimize'):
            print(HELP_OPTIMIZE)
            sys.exit(0)

        # load an existing model if so configured
        progress = self.kwargs.get('save_progress')
        if progress and os.path.exists(progress):
            self.load(progress)
        else:
            assert network_class, 'network class must be provided!'
            assert network_class is not feedforward.Network, \
                'use a concrete theanets.Network subclass ' \
                'like theanets.{Autoencoder,Regressor,...}'
            self.network = network_class(**self.kwargs)

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
        trainer : :class:`trainer.Trainer`
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

    def create_dataset(self, data, **kwargs):
        '''Create a dataset for this experiment.

        Parameters
        ----------
        data : ndarray, (ndarray, ndarray), or callable
            The values that you provide for data will be encapsulated inside a
            :class:`dataset.Dataset` instance; see that class for documentation
            on the types of things it needs. In particular, you can currently
            pass in either a list/array/etc. of data, or a callable that
            generates data dynamically.

        Returns
        -------
        data : :class:`dataset.Dataset`
            A dataset capable of providing mini-batches of data to a training
            algorithm.
        '''
        samples, labels = data, None
        if isinstance(data, (tuple, list)):
            if len(data) > 0:
                samples = data[0]
            if len(data) > 1:
                labels = data[1]
        name = kwargs.get('name', 'dataset')
        b, i, s = 'batch_size', 'iteration_size', '{}_batches'.format(name)
        return dataset.Dataset(
            samples, labels=labels, name=name,
            batch_size=kwargs.get(b, self.kwargs.get(b, 32)),
            iteration_size=kwargs.get(i, kwargs.get(s, self.kwargs.get(s))),
            axis=kwargs.get('axis'))

    def run(self, *args, **kwargs):
        warnings.warn(
            'please use Experiment.train() instead of Experiment.run()',
            DeprecationWarning)
        return self.train(*args, **kwargs)

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
        if optimize is None:
            optimize = self.kwargs.get('optimize')
        if not optimize:
            optimize = 'rmsprop'  # use rmsprop if nothing else is defined.
        if isinstance(optimize, str):
            optimize = optimize.split()

        # set up auto-saving if enabled
        progress = self.kwargs.get('save_progress')
        timeout = self.kwargs.get('save_every', 0)
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
        logging.info('saving model to %s', path)
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
        network : :class:`feedforward.Network`
            A newly-constructed network, with topology and parameters loaded
            from the given pickle file.
        '''
        logging.info('loading model from %s', path)
        self.network = feedforward.load(path, **kwargs)
        return self.network

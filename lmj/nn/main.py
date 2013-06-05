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

import lmj.cli
import theano.tensor as TT

from .dataset import SequenceDataset as Dataset
from . import trainer

logging = lmj.cli.get_logger(__name__)


def parse_args(**overrides):
    '''Parse command-line arguments, overriding with keyword arguments.

    Returns an ordered pair of the command-line argument structure, as well as a
    dictionary version of these arguments.
    '''
    args = lmj.cli.get_args().parse_args()
    for k, v in overrides.iteritems():
        setattr(args, k, v)
    kwargs = {}
    kwargs.update(vars(args))
    logging.info('runtime arguments:')
    for k in sorted(kwargs):
        logging.info('--%s = %s', k, kwargs[k])
    return args, kwargs


class Experiment(object):
    '''This class encapsulates tasks for training and evaluating a network.
    '''

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
        self.args, kwargs = parse_args(**overrides)
        self.network = self._build_network(network_class)
        self.datasets = {}
        self.trainer = self._build_trainer(**kwargs)

    def _build_network(self, network_class):
        '''Build a Network class instance to compute input transformations.
        '''
        activation = self._build_activation()
        if hasattr(activation, 'lmj_nn_name'):
            logging.info('activation: %s', activation.lmj_nn_name)

        return network_class(
            layers=self.args.layers,
            activation=activation,
            decode=self.args.decode,
            tied_weights=self.args.tied_weights,
            input_noise=self.args.input_noise,
            hidden_noise=self.args.hidden_noise,
            input_dropouts=self.args.input_dropouts,
            hidden_dropouts=self.args.hidden_dropouts,
            damping=self.args.damping,
            )

    def _build_activation(self, act=None):
        '''Given an activation description, return a callable that implements it.
        '''
        def compose(a, b):
            c = lambda z: b(a(z))
            c.lmj_nn_name = '%s(%s)' % (b.lmj_nn_name, a.lmj_nn_name)
            return c
        act = act or self.args.activation.lower()
        if '+' in act:
            return reduce(compose, (self._build_activation(a) for a in act.split('+')))
        options = {
            'tanh': TT.tanh,
            'linear': lambda z: z,
            'logistic': TT.nnet.sigmoid,

            # shorthands
            'relu': lambda z: TT.maximum(0, z),

            # modifiers
            'rect:max': lambda z: TT.minimum(1, z),
            'rect:min': lambda z: TT.maximum(0, z),

            # normalization
            'norm:dc': lambda z: z - z.mean(axis=1)[:, None],
            'norm:max': lambda z: z / TT.maximum(1e-10, abs(z).max(axis=1)[:, None]),
            'norm:std': lambda z: z / TT.maximum(1e-10, z.std(axis=1)[:, None]),
            }
        for k, v in options.iteritems():
            v.lmj_nn_name = k
        try:
            return options[act]
        except:
            raise KeyError('unknown --activation %s' % act)

    def _build_trainer(self, **kwargs):
        '''Build a Trainer class instance for adjusting network parameters.

        Keyword arguments are passed as-is to the underlying Trainer instance.
        '''
        trainer_class = self._build_trainer_class()
        return trainer_class(self.network, **kwargs)

    def _build_trainer_class(self, opt=None):
        '''Given a trainer description, build a trainer class that implements it.
        '''
        opt = opt or self.args.optimize.lower()
        if '+' in opt:
            return trainer.Cascaded(self._build_trainer_class(o) for o in opt.split('+'))
        try:
            return {
                'hf': trainer.HF,
                'sgd': trainer.SGD,
                'sample': trainer.Sample,
                'layerwise': trainer.Layerwise,
                'force': trainer.FORCE,
                }[opt]
        except:
            raise KeyError('unknown --optimize %s' % opt)

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
        if not isinstance(dataset, (tuple, list)):
            dataset = (dataset, )
        self.datasets[label] = Dataset(*dataset, **kw)

    def run(self, train=None, valid=None):
        '''Run this experiment by training (and validating) a network.

        Before calling this method, datasets will typically need to have been
        added to the experiment by calling add_dataset(...). However, as a
        shortcut, you can provide training and validation data as arguments to
        this method, and these arguments will be used to add datasets as needed.

        Usually the output of this method is whatever is logged to the console
        during training. After training completes, the network attribute of this
        class will contain the trained network parameters.
        '''
        if train is not None:
            if 'train' not in self.datasets:
                self.add_dataset('train', train)
            if 'cg' not in self.datasets:
                self.add_dataset('cg', train)
        if valid is not None and 'valid' not in self.datasets:
            self.add_dataset('valid', valid)
        self.trainer.train(self.datasets['train'],
                           valid=self.datasets['valid'],
                           cg_set=self.datasets['cg'])

    def save(self, path):
        '''Save the parameters in the network to a pickle file on disk.
        '''
        self.network.save(path)

    def load(self, path):
        '''Load the parameters in the network from a pickle file on disk.
        '''
        self.network.load(path)

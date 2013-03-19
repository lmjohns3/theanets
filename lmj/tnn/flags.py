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

'''This file contains command line flags and a main method.'''

import argparse
import sys
import theano.tensor as TT

from .dataset import SequenceDataset as Dataset
from . import log
from . import trainer

logging = log.get_logger(__name__)

class ArgParser(argparse.ArgumentParser):
    SANE_DEFAULTS = dict(
        fromfile_prefix_chars='@',
        conflict_handler='resolve',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    def __init__(self, *args, **kwargs):
        kwargs.update(ArgParser.SANE_DEFAULTS)
        super(ArgParser, self).__init__(*args, **kwargs)

    def convert_arg_line_to_args(self, line):
        '''Remove # comments and blank lines from arg files.'''
        line = line.split('#')[0].strip()
        if line:
            if line[0] == '-' and ' ' in line:
                for p in line.split():
                    yield p
            else:
                yield line

FLAGS = ArgParser()

g = FLAGS.add_argument_group('Architecture')
g.add_argument('-n', '--layers', nargs='+', type=int, metavar='N',
               help='construct a network with layers of size N1, N2, ...')
g.add_argument('-g', '--activation', default='logistic', metavar='[linear|logistic|tanh|relu]',
               help='function for hidden unit activations')
g.add_argument('-t', '--tied-weights', action='store_true',
               help='tie encoding and decoding weights')
g.add_argument('--decode', type=int, default=1, metavar='N',
               help='decode from the final N layers of the net')
g.add_argument('--damping', type=float, metavar='R',
               help='damp recurrent network with R in [0, 1]')

g = FLAGS.add_argument_group('Training')
g.add_argument('-O', '--optimize', default='sgd', metavar='[hf|layerwise|sgd|sample]',
               help='train with the given optimization method')
g.add_argument('--no-learn-biases', action='store_true',
               help='if set, do not update bias parameters during learning')
g.add_argument('-u', '--num-updates', type=int, default=128, metavar='N',
               help='perform at most N parameter updates')
g.add_argument('-p', '--patience', type=int, default=15, metavar='N',
               help='stop training if no improvement for N updates')
g.add_argument('-v', '--validate', type=int, default=3, metavar='N',
               help='validate the model every N updates')
g.add_argument('-b', '--batch-size', type=int, default=64, metavar='N',
               help='train with mini-batches of size N')
g.add_argument('-B', '--train-batches', type=int, metavar='N',
               help='use at most N batches during gradient computations')
g.add_argument('-V', '--valid-batches', type=int, metavar='N',
               help='use at most N batches during validation')

g = FLAGS.add_argument_group('Regularization')
g.add_argument('--input-noise', type=float, default=0, metavar='S',
               help='add noise to network inputs drawn from N(0, S)')
g.add_argument('--input-dropouts', type=float, default=0, metavar='R',
               help='randomly set fraction R of input activations to 0')
g.add_argument('--hidden-noise', type=float, default=0, metavar='S',
               help='add noise to hidden activations drawn from N(0, S)')
g.add_argument('--hidden-dropouts', type=float, default=0, metavar='R',
               help='randomly set fraction R of hidden activations to 0')
g.add_argument('--hidden-l1', type=float, default=0, metavar='K',
               help='regularize hidden activity with K on the L1 term')
g.add_argument('--hidden-l2', type=float, default=0, metavar='K',
               help='regularize hidden activity with K on the L2 term')
g.add_argument('--weight-l1', type=float, default=0, metavar='K',
               help='regularize network weights with K on the L1 term')
g.add_argument('--weight-l2', type=float, default=0, metavar='K',
               help='regularize network weights with K on the L2 term')

g = FLAGS.add_argument_group('SGD Optimization')
g.add_argument('-l', '--learning-rate', type=float, default=0.1, metavar='V',
               help='train the network with a learning rate of V')
g.add_argument('-d', '--decay', type=float, default=0.01, metavar='R',
               help='decay the learning rate by R each epoch')
g.add_argument('-m', '--momentum', type=float, default=0.1, metavar='V',
               help='train the network with momentum of V')
g.add_argument('--min-improvement', type=float, default=0.01, metavar='R',
               help='train until relative improvement is less than R')

g = FLAGS.add_argument_group('HF Optimization')
g.add_argument('-C', '--cg-batches', type=int, metavar='N',
               help='use at most N batches for CG computation')
g.add_argument('--initial-lambda', type=float, default=1., metavar='K',
               help='start the HF method with Tikhonov damping of K')
g.add_argument('--preconditioner', action='store_true',
               help='precondition the system during CG')
g.add_argument('--save-progress', metavar='FILE',
               help='save the model periodically to FILE')


class Main(object):
    '''This class sets up the infrastructure to train a net.

    Two methods must be implemented by subclasses -- get_network must return the
    Network subclass to instantiate, and get_datasets must return a tuple of
    training and validation datasets. Subclasses have access to command line
    arguments through self.args.
    '''

    def __init__(self, args=None, **kwargs):
        self.args = args or FLAGS.parse_args()
        for k, v in kwargs.iteritems():
            setattr(self.args, k, v)

        kwargs = {}
        kwargs.update(vars(self.args))
        logging.info('command-line options:')
        for k in sorted(kwargs):
            logging.info('--%s = %s', k, kwargs[k])

        activation = self.get_activation()
        if hasattr(activation, 'lmj_nn_name'):
            logging.info('activation: %s', activation.lmj_nn_name)

        self.net = self.get_network()(
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

        kw = dict(size=self.args.batch_size)
        train_, valid_ = tuple(self.get_datasets())[:2]
        if not isinstance(train_, (tuple, list)):
            train_ = (train_, )
        if not isinstance(valid_, (tuple, list)):
            valid_ = (valid_, )

        kw['batches'] = self.args.train_batches
        self.train_set = Dataset('train', *train_, **kw)

        kw['batches'] = self.args.valid_batches
        self.valid_set = Dataset('valid', *valid_, **kw)

        kw['batches'] = self.args.cg_batches
        kwargs['cg_set'] = Dataset('cg', *train_, **kw)

        self.trainer = self.get_trainer()(self.net, **kwargs)

    def train(self):
        self.trainer.train(self.train_set, self.valid_set)

    def get_activation(self, act=None):
        def compose(a, b):
            c = lambda z: b(a(z))
            c.lmj_nn_name = '%s(%s)' % (b.lmj_nn_name, a.lmj_nn_name)
            return c
        act = act or self.args.activation.lower()
        if '+' in act:
            return reduce(compose, (self.get_activation(a) for a in act.split('+')))
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


    def get_trainer(self, opt=None):
        opt = opt or self.args.optimize.lower()
        if '+' in opt:
            return trainer.Cascaded(self.get_trainer(o) for o in opt.split('+'))
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

    def get_network(self):
        raise NotImplementedError

    def get_datasets(self):
        raise NotImplementedError

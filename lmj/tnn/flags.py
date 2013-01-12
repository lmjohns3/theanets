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

import logging
import optparse
import sys
import theano.tensor as TT

from .dataset import SequenceDataset as Dataset
from . import trainer

FLAGS = optparse.OptionParser()

g = optparse.OptionGroup(FLAGS, 'Architecture')
g.add_option('', '--decode', type=int, default=1, metavar='N',
             help='decode from the final N layers of the net (1)')
g.add_option('-n', '--layers', metavar='N0,N1,...',
             help='construct a network with layers of size N0, N1, ...')
g.add_option('-g', '--activation', default='', metavar='[linear|relu|tanh|retanh]',
             help='use g(z) for hidden unit activations (logistic)')
g.add_option('-t', '--tied-weights', action='store_true',
             help='tie decoding weights to encoding weights')
g.add_option('-z', '--normalize', default='', metavar='[max]',
             help='normalize hidden unit activations')
g.add_option('', '--input-noise', type=float, default=0, metavar='S',
             help='add noise to network inputs drawn from N(0, S) (0)')
g.add_option('', '--hidden-noise', type=float, default=0, metavar='S',
             help='add noise to hidden activations drawn from N(0, S) (0)')
FLAGS.add_option_group(g)

g = optparse.OptionGroup(FLAGS, 'Training')
g.add_option('-O', '--optimize', default='sgd', metavar='[hf|sgd+hf]',
             help='train with the given optimization method (sgd)')
g.add_option('-v', '--validate', type=int, default=3, metavar='N',
             help='validate the model every N updates (3)')
g.add_option('-s', '--batch-size', type=int, default=64, metavar='N',
             help='split all data sets into batches of size N (64)')
g.add_option('-B', '--train-batches', type=int, metavar='N',
             help='use at most N batches during gradient computations')
g.add_option('-V', '--valid-batches', type=int, metavar='N',
             help='use at most N batches during validation')
g.add_option('', '--hidden-l1', type=float, metavar='K',
             help='regularize hidden activity with K on the L1 term')
g.add_option('', '--hidden-l2', type=float, metavar='K',
             help='regularize hidden activity with K on the L2 term')
g.add_option('', '--weight-l1', type=float, metavar='K',
             help='regularize network weights with K on the L1 term')
g.add_option('', '--weight-l2', type=float, metavar='K',
             help='regularize network weights with K on the L2 term')
g.add_option('', '--num-updates', type=int, default=128, metavar='N',
             help='perform at most N parameter updates (128)')
g.add_option('', '--patience', type=int, default=16, metavar='N',
             help='stop training if no improvement for N updates (16)')
FLAGS.add_option_group(g)

g = optparse.OptionGroup(FLAGS, 'SGD Optimization')
g.add_option('-d', '--decay', type=float, default=0.99, metavar='R',
             help='decay the learning rate by R each epoch (0.99)')
g.add_option('-l', '--learning-rate', type=float, default=0.1, metavar='R',
             help='train the network with a learning rate of R (0.1)')
g.add_option('', '--min-improvement', type=float, default=0, metavar='N',
             help='train until relative cost decrease is less than N (0)')
g.add_option('-m', '--momentum', type=float, default=0.1, metavar='R',
             help='train the network with momentum of R (0.1)')
FLAGS.add_option_group(g)

g = optparse.OptionGroup(FLAGS, 'HF Optimization')
g.add_option('', '--cg-batches', type=int, metavar='N',
             help='use at most N batches for CG computation')
g.add_option('', '--initial-lambda', type=float, default=1., metavar='K',
             help='start the HF method with Tikhonov damping of K (1.)')
g.add_option('', '--preconditioner', action='store_true',
             help='precondition the system during CG')
g.add_option('', '--save-progress', metavar='FILE',
             help='save the model periodically to FILE')
FLAGS.add_option_group(g)


class Main(object):
    '''This class sets up the infrastructure to train a net.

    Two methods must be implemented by subclasses -- get_network must return the
    Network subclass to instantiate, and get_datasets must return a tuple of
    training and validation datasets. Subclasses have access to self.opts
    (command line options) and self.args (command line arguments).
    '''

    def __init__(self):
        self.opts, self.args = FLAGS.parse_args()

        kwargs = eval(str(self.opts))
        logging.info('command-line options:')
        for k in sorted(kwargs):
            logging.info('--%s = %s', k, kwargs[k])

        self.net = self.get_network()(
            layers=eval(self.opts.layers),
            activation=self.get_activation(),
            decode=self.opts.decode,
            tied_weights=self.opts.tied_weights,
            input_noise=self.opts.input_noise,
            hidden_noise=self.opts.hidden_noise,
            )

        kw = dict(size=self.opts.batch_size)
        train_, valid_ = tuple(self.get_datasets())[:2]
        if not isinstance(train_, (tuple, list)):
            train_ = (train_, )
        if not isinstance(valid_, (tuple, list)):
            valid_ = (valid_, )

        kw['batches'] = self.opts.train_batches
        self.train_set = Dataset('train', *train_, **kw)

        kw['batches'] = self.opts.valid_batches
        self.valid_set = Dataset('valid', *valid_, **kw)

        kw['batches'] = self.opts.cg_batches
        kwargs['cg_set'] = Dataset('cg', *train_, **kw)

        self.trainer = self.get_trainer()(self.net, **kwargs)

    def train(self):
        self.trainer.train(self.train_set, self.valid_set)
        return self.net

    def get_activation(self):
        g = TT.nnet.sigmoid
        if self.opts.activation.lower() == 'tanh':
            g = TT.tanh
        if self.opts.activation.lower() == 'linear':
            g = lambda z: z
        if self.opts.activation.lower() == 'relu':
            g = lambda z: TT.maximum(0, z)
        if self.opts.activation.lower() == 'retanh':
            g = lambda z: TT.maximum(0, TT.tanh(z))

        f = g
        if self.opts.normalize.lower().startswith('m'):
            n = lambda z: z / TT.maximum(1e-10, abs(z).max(axis=1)[:, None])
            f = lambda z: n(g(z))

        return f

    def get_trainer(self):
        t = trainer.SGD
        O = (self.opts.optimize or '').lower()
        if O.startswith('h'):
            t = trainer.HF
        if '+' in O or O.startswith('c'):
            t = trainer.Cascaded
        return t

    def get_network(self):
        raise NotImplementedError

    def get_datasets(self):
        raise NotImplementedError

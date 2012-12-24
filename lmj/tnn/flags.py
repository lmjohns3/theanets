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
g.add_option('-n', '--layers', default='784,100,10', metavar='N0,N1,...',
             help='construct a network with layers of size N0, N1, ... (None)')
g.add_option('-g', '--nonlinearity', default='sigmoid', metavar='[sig|tanh|relu]',
             help='use the given nonlinearity for hidden unit activations (sig)')
FLAGS.add_option_group(g)

g = optparse.OptionGroup(FLAGS, 'Training')
g.add_option('-o', '--optimize', metavar='[sgd|hf|sgd+hf]',
             help='train with the given optimization method (sgd)')
g.add_option('-v', '--validate', type=int, default=3, metavar='N',
             help='validate the model every N updates (3)')
g.add_option('-s', '--batch-size', type=int, default=100, metavar='N',
             help='split all data sets into batches of size N (100)')
g.add_option('-b', '--train-batches', type=int, metavar='N',
             help='use at most N batches during gradient computations (None)')
g.add_option('-B', '--valid-batches', type=int, metavar='N',
             help='use at most N batches during validation (None)')
g.add_option('', '--test-batches', type=int, metavar='N',
             help='use at most N batches during testing (None)')
g.add_option('', '--activity-l1', type=float, default=0., metavar='K',
             help='regularize network activity with K on the L1 term (0.)')
g.add_option('', '--weight-l1', type=float, default=0., metavar='K',
             help='regularize network weights with K on the L1 term (0.)')
g.add_option('', '--weight-l2', type=float, default=0., metavar='K',
             help='regularize network weights with K on the L2 term (0.)')
FLAGS.add_option_group(g)

g = optparse.OptionGroup(FLAGS, 'SGD Optimization')
g.add_option('-d', '--decay', type=float, default=1., metavar='R',
             help='decay the learning rate by R each epoch (1.)')
g.add_option('-l', '--learning-rate', type=float, default=0.1, metavar='R',
             help='train the network with a learning rate of R (0.1)')
g.add_option('', '--min-improvement', type=float, default=1e-4, metavar='N',
             help='train until relative cost decrease is less than N (1e-4)')
g.add_option('-m', '--momentum', type=float, default=0.1, metavar='R',
             help='train the network with momentum of R (0.1)')
FLAGS.add_option_group(g)

g = optparse.OptionGroup(FLAGS, 'HF Optimization')
g.add_option('', '--cg-batches', type=int, metavar='N',
             help='use at most N batches for CG computation (None)')
g.add_option('', '--initial-lambda', type=float, default=1., metavar='K',
             help='start the HF method with Tikhonov damping of K (1.)')
g.add_option('', '--num-updates', type=int, default=100, metavar='N',
             help='perform at most N HF parameter updates (100)')
g.add_option('', '--patience', type=int, default=10, metavar='N',
             help='stop training if no improvement for N validations (10)')
g.add_option('', '--preconditioner', default=False, action='store_true',
             help='precondition the system during CG')
g.add_option('', '--save-progress', metavar='FILE',
             help='save the model periodically to FILE (None)')
FLAGS.add_option_group(g)


class Main(object):
    '''This class sets up the infrastructure to train a net.

    Two methods must be implemented by subclasses -- get_network must return the
    Network subclass to instantiate, and get_datasets must return a tuple of
    training, validation, and testing datasets. Subclasses have access to
    self.opts (command line options) and self.args (command line arguments).
    '''

    def __init__(self):
        self.opts, self.args = FLAGS.parse_args()

        kwargs = eval(str(self.opts))
        logging.info('command-line options:')
        for k in sorted(kwargs):
            logging.info('--%s = %s', k, kwargs[k])

        self.net = self.get_network()(
            eval(self.opts.layers),
            self.get_nonlinearity(self.opts),
            self.opts.decode)

        kw = dict(size=self.opts.batch_size)
        train_, valid_, test_ = self.get_datasets()

        kw['batches'] = self.opts.train_batches
        self.train_set = Dataset('train', *train_, **kw)

        kw['batches'] = self.opts.valid_batches
        self.valid_set = Dataset('valid', *valid_, **kw)

        kw['batches'] = self.opts.test_batches
        kwargs['test_set'] = Dataset('test', *test_, **kw)

        kw['batches'] = self.opts.cg_batches
        kwargs['cg_set'] = Dataset('cg', *train_, **kw)

        self.trainer = self.get_trainer(self.opts)(self.net, **kwargs)

    def train(self):
        self.trainer.train(self.train_set, self.valid_set)
        return self.net

    def get_nonlinearity(self, opts):
        if opts.nonlinearity.lower().startswith('r'):
            return lambda z: TT.maximum(0, z)
        if opts.nonlinearity.lower().startswith('s'):
            return TT.nnet.sigmoid
        return TT.tanh

    def get_trainer(self, opts):
        if opts.optimize.lower().startswith('h'):
            return trainer.HF
        if '+' in opts.optimize or opts.optimize.lower().startswith('c'):
            return trainer.Cascaded
        return trainer.SGD

    def get_network(self):
        raise NotImplementedError

    def get_datasets(self):
        raise NotImplementedError

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
FLAGS.add_option('', '--layers', default='784,100,10', metavar='N0,N1,...',
                 help='construct the network with layers of N0, N1, ... units')
FLAGS.add_option('', '--decode', type=int, default=1, metavar='N',
                 help='decode from the final N layers of the net')
FLAGS.add_option('', '--nonlinearity', default='sigmoid', metavar='[sig|tanh|relu]',
                 help='use the given nonlinearity for hidden unit activations')
FLAGS.add_option('', '--param-l1', type=float, default=0., metavar='K',
                 help='regularize network parameters with K on the L1 term')
FLAGS.add_option('', '--param-l2', type=float, default=0., metavar='K',
                 help='regularize network parameters with K on the L2 term')
FLAGS.add_option('', '--activity-l1', type=float, default=0., metavar='K',
                 help='regularize network activity with K on the L1 term')
FLAGS.add_option('', '--optimize', metavar='[sgd|hf|sgd+hf]',
                 help='use the given optimization method')

g = optparse.OptionGroup(FLAGS, 'Training')
g.add_option('', '--batch-size', type=int, default=100, metavar='N',
             help='split all data sets into batches of size N')
g.add_option('', '--train-batches', type=int, metavar='N',
             help='use at most N batches during gradient computations')
g.add_option('', '--valid-batches', type=int, metavar='N',
             help='use at most N batches during validation')
g.add_option('', '--min-improvement', type=float, default=1e-4, metavar='N',
             help='train until relative cost decrease is less than N')
g.add_option('', '--validate', type=int, default=3, metavar='N',
             help='validate the model every N updates')
g.add_option('', '--save-progress', metavar='FILE',
             help='save the model periodically to FILE')
FLAGS.add_option_group(g)

g = optparse.OptionGroup(FLAGS, 'SGD Optimization')
g.add_option('', '--decay', type=float, default=1., metavar='R',
             help='train the network with a learning rate of R')
g.add_option('', '--learning-rate', type=float, default=0.1, metavar='R',
             help='train the network with a learning rate of R')
g.add_option('', '--momentum', type=float, default=0.1, metavar='R',
             help='train the network with momentum of R')
FLAGS.add_option_group(g)

g = optparse.OptionGroup(FLAGS, 'HF Optimization')
g.add_option('', '--cg-batches', type=int, metavar='N',
             help='use at most N batches for CG computation')
g.add_option('', '--patience', type=int, default=10, metavar='N',
             help='stop training if no improvement for N validations')
g.add_option('', '--initial-lambda', type=float, default=1., metavar='K',
             help='start CG with a lambda of K')
FLAGS.add_option_group(g)


def main(Network, get_datasets):
    '''This main function is a generic TODO for training a neural net.'''
    opts, args = FLAGS.parse_args()

    kwargs = eval(str(opts))

    logging.info('command-line options:')
    for k in sorted(kwargs):
        logging.info('--%s = %s', k, kwargs[k])

    hidden_nonlin = TT.tanh
    if opts.nonlinearity.lower().startswith('r'):
        hidden_nonlin = lambda z: TT.maximum(0, z)
    if opts.nonlinearity.lower().startswith('s'):
        hidden_nonlin = TT.nnet.sigmoid
    net = Network(eval(opts.layers), hidden_nonlin, opts.decode)

    train_, valid_, test_ = get_datasets(opts, args)
    train_set = Dataset('train', *train_, size=opts.batch_size)
    valid_set = Dataset('valid', *valid_, size=opts.batch_size)
    kwargs['test_set'] = Dataset('test', *test_, size=opts.batch_size)
    kwargs['cg_set'] = Dataset('cg', *train_, size=opts.batch_size, batches=opts.cg_batches)

    Trainer = {'hf': trainer.HF, 'sgd+hf': trainer.Cascaded}.get(opts.optimize, trainer.SGD)
    Trainer(net, **kwargs).train(train_set, valid_set)

    return net

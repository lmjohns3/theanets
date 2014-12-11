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

'''This file contains command line flags.'''

import argparse
import climate

climate.add_arg('--help-activation', action='store_true',
                help='show available activation functions')
climate.add_arg('--help-optimize', action='store_true',
                help='show available optimization algorithms')

g = climate.add_arg_group('Architecture')
g.add_argument('-n', '--layers', nargs='+', type=int, metavar='N',
               help='construct a network with layers of size N1, N2, ...')
g.add_argument('-g', '--hidden-activation', default='logistic', metavar='FUNC',
               help='function for hidden unit activations')
g.add_argument('--output-activation', default='linear', metavar='FUNC',
               help='function for output unit activations')
g.add_argument('-t', '--tied-weights', action='store_true',
               help='tie encoding and decoding weights')
g.add_argument('--decode', type=int, default=1, metavar='N',
               help='decode from the final N layers of the net')

g = climate.add_arg_group('Training')
g.add_argument('-O', '--optimize', default=(), nargs='+', metavar='ALGO',
               help='train with the given optimization algorithm(s)')
g.add_argument('--no-learn-biases', action='store_true',
               help='if set, do not update bias parameters during learning')
g.add_argument('--num-updates', type=int, default=10000, metavar='N',
               help='perform at most N HF/scipy parameter updates')
g.add_argument('-p', '--patience', type=int, default=50, metavar='N',
               help='stop SGD/HF training if no improvement for N updates')
g.add_argument('-v', '--validate', type=int, default=10, metavar='N',
               help='validate the model every N updates')
g.add_argument('-b', '--batch-size', type=int, default=64, metavar='N',
               help='train with mini-batches of size N')
g.add_argument('-B', '--train-batches', type=int, metavar='N',
               help='use at most N batches during gradient computations')
g.add_argument('-V', '--valid-batches', type=int, metavar='N',
               help='use at most N batches during validation')
g.add_argument('--save-progress', metavar='FILE',
               help='save the model periodically to FILE')
g.add_argument('--save-every', type=float, default=0, metavar='N',
               help='save the model every N iterations or -N minutes')

g = climate.add_arg_group('Regularization')
g.add_argument('--contractive-l2', type=float, default=0, metavar='S',
               help='penalize the Frobenius norm of the hidden Jacobian')
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

g = climate.add_arg_group('SGD-Based Optimization')
g.add_argument('-l', '--learning-rate', type=float, default=1e-4, metavar='V',
               help='train the network with a learning rate of V')
g.add_argument('-m', '--momentum', type=float, default=0.9, metavar='V',
               help='train the network with momentum of V')
g.add_argument('--min-improvement', type=float, default=0.01, metavar='R',
               help='train until relative improvement is less than R')

g = climate.add_arg_group('RmsProp Optimization')
g.add_argument('--rms-clip', type=float, default=1000, metavar='V',
               help='clip gradient values outside [-V, V]')
g.add_argument('--rms-ema', type=float, default=0.9, metavar='V',
               help='use an exponential moving average with weight V')

g = climate.add_arg_group('Rprop Optimization')
g.add_argument('--rprop-increase', type=float, default=1.01, metavar='R',
               help='increase parameter steps at rate R')
g.add_argument('--rprop-decrease', type=float, default=0.99, metavar='R',
               help='decrease parameter steps at rate R')
g.add_argument('--rprop-min-step', type=float, default=0., metavar='V',
               help='cap parameter steps to V at the smallest')
g.add_argument('--rprop-max-step', type=float, default=1., metavar='V',
               help='cap parameter steps to V at the largest')

g = climate.add_arg_group('HF Optimization')
g.add_argument('-C', '--cg-batches', type=int, metavar='N',
               help='use at most N batches for CG computation')
g.add_argument('--initial-lambda', type=float, default=1., metavar='K',
               help='start the HF method with Tikhonov damping of K')
g.add_argument('--global-backtracking', action='store_true',
               help='backtrack to lowest cost parameters during CG')
g.add_argument('--preconditioner', action='store_true',
               help='precondition the system during CG')

g = climate.add_arg_group('Recurrent Nets')
g.add_argument('--recurrent-error-start', type=int, default=3, metavar='T',
               help='compute network error starting at time T')
g.add_argument('--recurrent-radius', type=float, default=1.1, metavar='R',
               help='create recurrent weights with spectral radius R')
g.add_argument('--recurrent-sparsity', type=float, default=0, metavar='R',
               help='create recurrent weights with fraction R as 0')

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

g = climate.add_arg_group('Architecture')
g.add_argument('-n', '--layers', nargs='+', type=int, metavar='N',
               help='construct a network with layers of size N1, N2, ...')
g.add_argument('-g', '--activation', default='logistic', metavar='[linear|logistic|tanh|relu]',
               help='function for hidden unit activations')
g.add_argument('-t', '--tied-weights', action='store_true',
               help='tie encoding and decoding weights')
g.add_argument('--decode', type=int, default=1, metavar='N',
               help='decode from the final N layers of the net')

g = climate.add_arg_group('Training')
g.add_argument('-O', '--optimize', default=['sgd'], nargs='+', metavar='[hf|layerwise|sgd|sample]',
               help='train with the given optimization method(s)')
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
g.add_argument('--save-progress', metavar='FILE',
               help='save the model periodically to FILE')

g = climate.add_arg_group('Regularization')
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

g = climate.add_arg_group('SGD Optimization')
g.add_argument('-l', '--learning-rate', type=float, default=0.01, metavar='V',
               help='train the network with a learning rate of V')
g.add_argument('-d', '--learning-rate-decay', type=float, default=0.25, metavar='R',
               help='decay the learning rate by R after stagnant validations')
g.add_argument('-m', '--momentum', type=float, default=0, metavar='V',
               help='train the network with momentum of V')
g.add_argument('--min-improvement', type=float, default=0.01, metavar='R',
               help='train until relative improvement is less than R')
g.add_argument('--gradient-clip', type=float, default=1e5, metavar='R',
               help='clip elementwise gradients to [-R, R]')

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
g.add_argument('--pool-damping', type=float, default=0, metavar='R',
               help='damp recurrent network with R in [0, 1]')
g.add_argument('--pool-noise', type=float, default=0, metavar='S',
               help='add noise to recurrent units drawn from N(0, S)')
g.add_argument('--pool-dropouts', type=float, default=0, metavar='R',
               help='randomly set fraction R of recurrent units to 0')
g.add_argument('--pool-error-start', type=int, default=3, metavar='T',
               help='compute network error starting at time T')

'''This module contains command line flags.'''

import climate

climate.add_arg('--help-activation', action='store_true',
                help='show available activation functions')
climate.add_arg('--help-optimize', action='store_true',
                help='show available optimization algorithms')

g = climate.add_group('Architecture')
g.add_argument('-n', '--layers', nargs='+', type=int, metavar='N',
               help='construct a network with layers of size N1, N2, ...')
g.add_argument('-g', '--hidden-activation', default='logistic', metavar='FUNC',
               help='function for hidden unit activations')
g.add_argument('--output-activation', default='linear', metavar='FUNC',
               help='function for output unit activations')
g.add_argument('-t', '--tied-weights', action='store_true',
               help='tie encoding and decoding weights')
g.add_argument('--decode-from', type=int, default=1, metavar='N',
               help='decode from the final N layers of the net')

g = climate.add_group('Training')
g.add_argument('-O', '--optimize', default=(), nargs='+', metavar='ALGO',
               help='train with the given optimization algorithm(s)')
g.add_argument('-p', '--patience', type=int, default=4, metavar='N',
               help='stop training if less than --min-improvement for N validations')
g.add_argument('-v', '--validate-every', type=int, default=10, metavar='N',
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

g = climate.add_group('Regularization')
g.add_argument('--contractive', type=float, default=0, metavar='S',
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

g = climate.add_group('SGD-Based Optimization')
g.add_argument('-l', '--learning-rate', type=float, default=1e-4, metavar='V',
               help='train the network with a learning rate of V')
g.add_argument('-m', '--momentum', type=float, default=0.9, metavar='V',
               help='train the network with momentum of V')
g.add_argument('--min-improvement', type=float, default=0.01, metavar='R',
               help='train until relative improvement is less than R')
g.add_argument('--gradient-clip', type=float, default=1e6, metavar='V',
               help='clip elementwise gradient values outside [-V, V]')
g.add_argument('--max-gradient-norm', type=float, default=1e6, metavar='V',
               help='clip gradients with norms outside [-V, V]')

g = climate.add_group('RmsProp Optimization')
g.add_argument('--rms-halflife', type=float, default=7, metavar='N',
               help='use a half-life of N for RMS exponential moving averages')

g = climate.add_group('Rprop Optimization')
g.add_argument('--rprop-increase', type=float, default=1.01, metavar='R',
               help='increase parameter steps at rate R')
g.add_argument('--rprop-decrease', type=float, default=0.99, metavar='R',
               help='decrease parameter steps at rate R')
g.add_argument('--rprop-min-step', type=float, default=0., metavar='V',
               help='cap parameter steps to V at the smallest')
g.add_argument('--rprop-max-step', type=float, default=1., metavar='V',
               help='cap parameter steps to V at the largest')

g = climate.add_group('HF Optimization')
g.add_argument('-C', '--cg-batches', type=int, metavar='N',
               help='use at most N batches for CG computation')
g.add_argument('--initial-lambda', type=float, default=1., metavar='K',
               help='start the HF method with Tikhonov damping of K')
g.add_argument('--global-backtracking', action='store_true',
               help='backtrack to lowest cost parameters during CG')
g.add_argument('--preconditioner', action='store_true',
               help='precondition the system during CG')

g = climate.add_group('Recurrent Nets')
g.add_argument('--recurrent-error-start', type=int, default=3, metavar='T',
               help='compute recurrent network error starting at time T')

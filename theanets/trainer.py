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

'''This file contains optimization methods for neural networks.'''

import climate
import collections
import itertools
import numpy as np
import numpy.random as rng
import theano
import theano.tensor as TT
import sys

from . import dataset
from . import feedforward
from . import recurrent

logging = climate.get_logger(__name__)


class Trainer(object):
    '''This is a base class for all trainers.'''

    def update_params(self, targets):
        for param, target in zip(self.params, targets):
            param.set_value(target)


class SGD(Trainer):
    '''Stochastic gradient descent network trainer.'''

    def __init__(self, network, **kwargs):
        self.params = network.params(**kwargs)
        self.validation_frequency = kwargs.get('validate', 3)
        self.min_improvement = kwargs.get('min_improvement', 0.)
        self.iterations = kwargs.get('num_updates', 1000)
        self.patience = kwargs.get('patience', 100)
        self.momentum = kwargs.get('momentum', 0.)
        self.grad_clip = kwargs.get('gradient_clip', 1e5)
        self.learning_rate = kwargs.get('learning_rate', 0.1)
        self.learning_rate_decay = kwargs.get('learning_rate_decay', 0.1)

        J = network.J(**kwargs)
        costs = [J]
        self.cost_names = ['J']
        for name, monitor in network.monitors:
            self.cost_names.append(name)
            costs.append(monitor)
        self.f_train = theano.function(network.inputs, costs, updates=network.updates)
        self.f_grad = theano.function(network.inputs, [TT.grad(J, p) for p in self.params])
        self.f_eval = theano.function(network.inputs, costs)

    def train(self, train_set, valid_set=None, **kwargs):
        '''We train over mini-batches and evaluate periodically.'''
        best_cost = 1e100
        best_iter = 0
        best_params = []
        velocities = []
        P = len(self.params)
        for p in self.params:
            v = p.get_value()
            best_params.append(v.copy())
            velocities.append(np.zeros_like(v))

        learn = self._nag if self.momentum > 0 else self._sgd

        for i in xrange(self.iterations):
            # if it's time, evaluate the model on the validation dataset.
            if not i % self.validation_frequency:
                try:
                    costs = np.mean([self.f_eval(*x) for x in valid_set], axis=0)
                except KeyboardInterrupt:
                    logging.info('interrupted !')
                    break
                marker = ''
                if (best_cost - costs[0]) / best_cost > self.min_improvement:
                    best_cost = costs[0]
                    best_iter = i
                    best_params = [p.get_value().copy() for p in self.params]
                    marker = ' *'
                else:
                    self.learning_rate *= 1 - self.learning_rate_decay

                cost_desc = ' '.join(
                    '%s=%.4f' % i for i in zip(self.cost_names, costs))
                logging.info('SGD %i -- valid %s%s', i + 1, cost_desc, marker)

                if i - best_iter > self.patience:
                    logging.error('patience elapsed, bailing out')
                    break

            costs = []
            grads = []
            try:
                for costs_, grads_ in learn(train_set, velocities):
                    costs.append(costs_)
                    grads.append(grads_)
            except KeyboardInterrupt:
                logging.info('interrupted !')
                break

            cost_desc = ' '.join(
                '%s=%.4f' % i for i in
                zip(self.cost_names, np.mean(costs, axis=0)))
            grad_desc = ' '.join(
                '%s=%.4f' % (p.name, x) for p, x in
                zip(self.params, np.mean(grads, axis=0)))
            logging.info('SGD %i/%i @%.2e -- train %s -- grad %s',
                         i + 1, self.iterations, self.learning_rate,
                         cost_desc, grad_desc)

        self.update_params(best_params)

    def _nag(self, train_set, velocities):
        '''Make one run through the training set.

        We update parameters after each minibatch according to Nesterov's
        Accelerated Gradient. The basic difference between NAG and "classical"
        momentum is that NAG computes the gradients at the position in parameter
        space where "classical" momentum would put us at the next step.

        In theory, this helps correct for oversteps during learning. If momentum
        would lead us to overshoot, then the gradient at that overshot place
        will point backwards, toward where we came from.
        '''
        gc = self.grad_clip
        # TODO: run this loop in parallel !
        for x in train_set:
            grads = []
            moves = []
            # first, move to the position in parameter space that we would get
            # to using classical momentum-based sgd.
            for param, vel in zip(self.params, velocities):
                u = self.momentum * vel
                v = param.get_value(borrow=True)
                v += u
                param.set_value(v, borrow=True)
                moves.append(u)
            for param, vel, grad, u in zip(self.params, velocities, self.f_grad(*x), moves):
                # measure the gradient at this new position.
                g = np.asarray(grad)
                grads.append(np.linalg.norm(g))
                # update the velocity using the new gradient. remember that
                # u = self.momentum * vel.
                np.clip(u - self.learning_rate * g, -gc, gc, out=vel)
                # subtract out the movement from momentum that we added in
                # above, and add the updated velocity.
                v = param.get_value(borrow=True)
                v += vel - u
                param.set_value(v, borrow=True)
            yield self.f_train(*x), grads

    def _sgd(self, train_set, velocities):
        '''Make one run through the training set.

        We update parameters after each minibatch according to standard
        stochastic gradient descent.
        '''
        # TODO: run this loop in parallel !
        for x in train_set:
            grads = []
            for param, grad in zip(self.params, self.f_grad(*x)):
                g = np.asarray(grad)
                grads.append(np.linalg.norm(g))
                v = param.get_value(borrow=True)
                v -= self.learning_rate * g
                param.set_value(v, borrow=True)
            yield self.f_train(*x), grads


class CG(Trainer):
    '''Conjugate gradient trainer for neural networks.'''

    def __init__(self, network, **kwargs):
        raise NotImplementedError


class LM(Trainer):
    '''Levenberg-Marquardt trainer for neural networks.

    Based on the description of the algorithm in "Levenberg-Marquardt
    Optimization" by Sam Roweis.
    '''

    def __init__(self, network, **kwargs):
        raise NotImplementedError


class HF(Trainer):
    '''The hessian free trainer shells out to an external implementation.

    hf.py was implemented by Nicholas Boulanger-Lewandowski and made available
    to the public (yay !). If you don't have a copy of the module handy, this
    class will attempt to download it from github.
    '''

    URL = 'https://raw.github.com/boulanni/theano-hf/master/hf.py'

    def __init__(self, network, **kwargs):
        import os, tempfile, urllib
        sys.path.append(tempfile.gettempdir())

        try:
            import hf
        except:
            # if hf failed to import, try downloading it and saving it locally.
            logging.error('hf import failed, attempting to download %s', HF.URL)
            path = os.path.join(tempfile.gettempdir(), 'hf.py')
            urllib.urlretrieve(HF.URL, path)
            logging.error('downloaded hf code to %s', path)
            import hf

        self.params = network.params(**kwargs)
        self.opt = hf.hf_optimizer(
            self.params,
            network.inputs,
            network.y,
            [network.J(**kwargs)] + [mon for _, mon in network.monitors],
            network.hiddens[-1] if isinstance(network, recurrent.Network) else None)

        # fix mapping from kwargs into a dict to send to the hf optimizer
        kwargs['validation_frequency'] = kwargs.pop('validate', sys.maxint)
        for k in set(kwargs) - set(self.opt.train.im_func.func_code.co_varnames[1:]):
            kwargs.pop(k)
        self.kwargs = kwargs

    def train(self, train_set, valid_set=None, **kwargs):
        self.update_params(self.opt.train(
            train_set, kwargs['cg_set'], validation=valid_set, **self.kwargs))


class Sample(Trainer):
    '''This trainer replaces network weights with samples from the input.'''

    @staticmethod
    def reservoir(xs, n):
        '''Select a random sample of n items from xs.'''
        pool = []
        for i, x in enumerate(xs):
            if len(pool) < n:
                pool.append(x / np.linalg.norm(x))
                continue
            j = rng.randint(i + 1)
            if j < n:
                pool[j] = x / np.linalg.norm(x)
        # if the pool still has fewer than n items, pad with distorted random
        # duplicates from the source data.
        L = len(pool)
        while len(pool) < n:
            x = pool[rng.randint(L)]
            pool.append(x + np.std(pool, axis=0) * rng.randn(*x.shape))
        return pool

    def __init__(self, network, **kwargs):
        self.network = network

    def train(self, train_set, valid_set=None, **kwargs):
        ifci = itertools.chain.from_iterable

        # set output (decoding) weights on the network.
        last = lambda x: x[-1] if isinstance(x, (tuple, list)) else x
        samples = ifci(last(t) for t in train_set)
        for w in self.network.weights:
            k, n = w.get_value(borrow=True).shape
            if w.name.startswith('W_out_'):
                arr = np.vstack(Sample.reservoir(samples, k))
                logging.info('setting weights for %s: %d x %d <- %s', w.name, k, n, arr.shape)
                w.set_value(arr)

        # set input (encoding) weights on the network.
        first = lambda x: x[0] if isinstance(x, (tuple, list)) else x
        samples = ifci(first(t) for t in train_set)
        for i, h in enumerate(self.network.hiddens):
            w = self.network.weights[i]
            m, k = w.get_value(borrow=True).shape
            arr = np.vstack(Sample.reservoir(samples, k)).T
            logging.info('setting weights for %s: %d x %d <- %s', w.name, m, k, arr.shape)
            w.set_value(arr)
            samples = ifci(self.network.feed_forward(first(t))[i-1] for t in train_set)


class Layerwise(Trainer):
    '''This trainer adapts parameters using a variant of layerwise pretraining.

    In this variant, we create "taps" at increasing depths into the original
    network weights, training only those weights that are below the tap. So, for
    a hypothetical binary classifier network with layers [3, 4, 5, 6, 2], we
    would first insert a tap after the first hidden layer (effectively a binary
    classifier in a [3, 4, 2] configuration) and train just that network. Then
    we insert a tap at the next layer (effectively training a [3, 4, 5, 2]
    classifier), and so forth.

    By inserting taps into the original network, we preserve all of the relevant
    settings of noise, dropouts, loss function and the like, in addition to
    obviating the need for copying trained weights around between different
    Network instances.
    '''

    def __init__(self, network, **kwargs):
        self.network = network
        self.kwargs = kwargs

    def train(self, train_set, valid_set=None, **kwargs):
        y = self.network.y
        hiddens = list(self.network.hiddens)
        weights = list(self.network.weights)
        biases = list(self.network.biases)

        nout = len(biases[-1].get_value(borrow=True))
        nhids = [len(b.get_value(borrow=True)) for b in biases]
        for i in range(1, len(nhids)):
            W, b, _ = self.network._weights_and_bias(nhids[i-1], nout, 'lwout-%d' % i)
            self.network.y = TT.dot(hiddens[i-1], W) + b
            self.network.hiddens = hiddens[:i]
            self.network.weights = weights[:i] + [W]
            self.network.biases = biases[:i] + [b]
            SGD(self.network, **self.kwargs).train(train_set, valid_set)
            self.network.save('/tmp/layerwise-%s-h%f-n%f-d%f-w%f-%d.pkl.gz' % (
                    ','.join(map(str, self.kwargs['layers'])),
                    self.kwargs['hidden_l1'],
                    self.kwargs['input_noise'],
                    self.kwargs['hidden_dropouts'],
                    self.kwargs['weight_l1'],
                    i))

        self.network.y = y
        self.network.hiddens = hiddens
        self.network.weights = weights
        self.network.biases = biases


class FORCE(Trainer):
    '''FORCE is a training method for recurrent nets by Sussillo & Abbott.

    This implementation needs some more love before it will work.
    '''

    def __init__(self, network, **kwargs):
        W_in, W_pool, W_out = network.weights

        n = W_pool.get_value(borrow=True).shape[0]
        self.alpha = kwargs.get('learning_rate', 1. / n)
        P = theano.shared(np.eye(n).astype(FLOAT) * self.alpha)

        k = TT.dot(P, network.state)
        rPr = TT.dot(network.state, k)
        c = 1. / (1. + rPr)
        dw = network.error(**kwargs) * c * k

        J = network.J(**kwargs)
        updates = {}
        updates[P] = P - c * TT.outer(k, k)
        updates[W_pool] = W_pool - dw
        updates[W_out] = W_out - dw
        updates[b_out] = b_out - self.alpha * TT.grad(J, b_out)

        costs = [J] + network.monitors
        self.f_eval = theano.function(network.inputs, costs)
        self.f_train = theano.function(network.inputs, costs, updates=updates)

    @property
    def learning_rate(self):
        return self.alpha

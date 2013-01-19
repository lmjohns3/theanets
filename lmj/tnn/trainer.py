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

import itertools
import logging
import numpy as np
import numpy.random as rng
import tempfile
import theano
import theano.tensor as TT
import sys


class Trainer(object):
    '''This is a base class for all trainers.'''

    def train(self, train_set, valid_set=None):
        '''By default, we train in iterations and evaluate periodically.'''
        best_cost = 1e100
        best_iter = 0
        best_params = [p.get_value().copy() for p in self.params]
        for i in xrange(self.iterations):
            if i - best_iter > self.patience:
                logging.error('patience elapsed, bailing out')
                break
            try:
                fmt = 'epoch %i[%.2g]: train %s'
                args = (i + 1,
                        self.learning_rate,
                        np.mean([self.f_train(*x) for x in train_set], axis=0),
                        )
                if (i + 1) % self.validation_frequency == 0:
                    metrics = np.mean([self.f_eval(*x) for x in valid_set], axis=0)
                    fmt += ' valid %s'
                    args += (metrics, )
                    if (best_cost - metrics[0]) / best_cost > self.min_improvement:
                        best_cost = metrics[0]
                        best_iter = i
                        best_params = [p.get_value().copy() for p in self.params]
                        fmt += ' *'
                self.finish_iteration()
            except KeyboardInterrupt:
                logging.info('interrupted !')
                break
            logging.info(fmt, *args)
        self.update_params(best_params)

    def update_params(self, targets):
        for param, target in zip(self.params, targets):
            param.set_value(target)

    def finish_iteration(self):
        pass

    def evaluate(self, test_set):
        return np.mean([self.f_eval(*i) for i in test_set], axis=0)


class SGD(Trainer):
    '''Stochastic gradient descent network trainer.'''

    def __init__(self, network, **kwargs):
        self.params = network.params(**kwargs)
        self.validation_frequency = kwargs.get('validate', 3)
        self.min_improvement = kwargs.get('min_improvement', 0.)
        self.iterations = kwargs.get('num_updates', 1e100)
        self.patience = kwargs.get('patience', 1e100)
        logging.info('%d named parameters to learn', len(self.params))

        decay = kwargs.get('decay', 0.01)
        m = kwargs.get('momentum', 0.1)
        lr = kwargs.get('learning_rate', 0.1)

        J = network.J(**kwargs)
        t = theano.shared(np.cast['float32'](0), name='t')
        updates = {}
        for param in self.params:
            grad = TT.grad(J, param)
            heading = theano.shared(
                np.zeros_like(param.get_value(borrow=True)),
                name='grad_%s' % param.name)
            updates[param] = param + heading
            updates[heading] = m * heading - lr * ((1 - decay) ** t) * grad

        costs = [J] + network.monitors
        self.f_eval = theano.function(network.inputs, costs)
        self.f_train = theano.function(network.inputs, costs, updates=updates)
        self.f_rate = theano.function([], [lr * ((1 - decay) ** t)])
        self.f_finish = theano.function([], [t], updates={t: t + 1})
        #theano.printing.pydotprint(
        #    theano.function(network.inputs, [J]), '/tmp/theano-network.png')

    @property
    def learning_rate(self):
        return self.f_rate()[0]

    def finish_iteration(self):
        self.f_finish()


class HF(Trainer):
    '''The hessian free trainer shells out to an external implementation.

    hf.py was implemented by Nicholas Boulanger-Lewandowski and made available
    to the public (yay !). If you don't have a copy of the module handy, this
    class will attempt to download it from github.
    '''

    URL = 'https://raw.github.com/boulanni/theano-hf/master/hf.py'

    def __init__(self, network, **kwargs):
        sys.path.append(tempfile.gettempdir())
        try:
            import hf
        except:
            # if hf failed to import, try downloading it and saving it locally.
            import os, urllib
            logging.error('hf import failed, attempting to download %s', HF.URL)
            path = os.path.join(tempfile.gettempdir(), 'hf.py')
            urllib.urlretrieve(HF.URL, path)
            logging.error('downloaded hf code to %s', path)
            del os
            del urllib
            import hf

        c = [network.J(**kwargs)] + network.monitors
        self.f_eval = theano.function(network.inputs, c)
        self.cg_set = kwargs.pop('cg_set')
        self.params = network.params(**kwargs)
        logging.info('%d parameter updates during training', len(self.params))
        self.opt = hf.hf_optimizer(self.params, network.inputs, network.y, c)

        # fix mapping from kwargs into a dict to send to the hf optimizer
        kwargs['validation_frequency'] = kwargs.pop('validate', sys.maxint)
        for k in set(kwargs) - set(self.opt.train.im_func.func_code.co_varnames[1:]):
            kwargs.pop(k)
        self.kwargs = kwargs

    def train(self, train_set, valid_set=None):
        self.update_params(self.opt.train(
            train_set, self.cg_set, validation=valid_set, **self.kwargs))


class Cascaded(Trainer):
    '''This trainer allows running multiple trainers sequentially.'''

    def __init__(self, trainers):
        self.trainers = trainers

    def __call__(self, network, **kwargs):
        self.trainers = (t(network, **kwargs) for t in self.trainers)
        return self

    def train(self, train_set, valid_set=None):
        for trainer in self.trainers:
            trainer.train(train_set, valid_set)


def reservoir(xs, n):
    '''Select a random sample of n items from xs.'''
    pool = []
    for i, x in enumerate(xs):
        if len(pool) < n:
            pool.append(x / np.linalg.norm(x))
            continue
        if n * rng.random() > i:
            pool[rng.randint(n)] = x / np.linalg.norm(x)
    return pool


class Data(Trainer):
    '''This trainer replaces network weights with samples from the input.'''

    def __init__(self, network, **kwargs):
        self.network = network

    def train(self, train_set, valid_set=None):
        ifci = itertools.chain.from_iterable
        first = lambda x: x[0] if isinstance(x, (tuple, list)) else x
        samples = ifci(first(t) for t in train_set)
        for i, h in enumerate(self.network.hiddens):
            w = self.network.weights[i]
            m, k = w.get_value(borrow=True).shape
            logging.info('setting weights for layer %d: %d x %d', i + 1, m, k)
            w.set_value(np.vstack(reservoir(samples, k)).T)
            samples = ifci(self.network(first(t))[i-1] for t in train_set)


class FORCE(Trainer):
    '''FORCE is a training method for recurrent nets by Sussillo & Abbott.'''

    def __init__(self, network, **kwargs):
        W_in, W_pool, W_out = network.weights

        n = W_pool.get_value(borrow=True).shape[0]
        self.alpha = kwargs.get('learning_rate', 1. / n)
        P = theano.shared(np.eye(n).astype(FLOAT) * self.alpha)

        k = TT.dot(P, network.state)
        rPr = TT.dot(network.state, k)
        c = 1. / (1. + rPr)
        dw = network.error(**kwargs) * c * k

        updates = {}
        updates[P] = P - c * TT.outer(k, k)
        updates[W_pool] = W_pool - dw
        updates[W_out] = W_out - dw
        #updates[b_out] = b_out - self.alpha * TT.grad(J, b_out)

        costs = [J] + network.monitors
        self.f_eval = theano.function(network.inputs, costs)
        self.f_train = theano.function(network.inputs, costs, updates=updates)

    @property
    def learning_rate(self):
        return self.alpha

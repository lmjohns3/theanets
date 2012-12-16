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

import logging
import numpy
import theano
import theano.tensor as TT
import sys


class Trainer(object):
    '''This is an abstract base class for all trainers.'''

    def train(self, train_set, valid_set=None):
        raise NotImplementedError

    def evaluate(self, test_set):
        return numpy.mean([self.f_eval(*i) for i in test_set], axis=0)


class SGD(Trainer):
    '''Stochastic gradient descent network trainer.'''

    def __init__(self, network, **kwargs):
        self.validation_frequency = kwargs.get('validate', 3)
        self.min_improvement = kwargs.get('min_improvement', 1e-4)

        decay = kwargs.get('decay', 1.)
        m = kwargs.get('momentum', 0.)
        lr = kwargs.get('learning_rate', 0.1)

        J = network.J(**kwargs)
        t = theano.shared(numpy.cast['float32'](0), name='t')
        updates = {t: t + 1}
        for param in network.params:
            grad = TT.grad(J, param)
            heading = theano.shared(
                numpy.zeros_like(param.get_value(borrow=True)),
                name='g_%s' % param.name)
            updates[param] = param + heading
            updates[heading] = m * heading - lr * (decay ** t) * grad

        costs = [J] + network.monitors
        self.f_eval = theano.function(network.inputs, costs)
        self.f_rate = theano.function([], [lr * (decay ** t)])
        self.f_train = theano.function(network.inputs, costs, updates=updates)
        #theano.printing.pydotprint(
        #    theano.function(network.inputs, [J]), '/tmp/theano-network.png')

    def train(self, train_set, valid_set=None):
        prev_cost = 1e101
        cost = 1e100
        iter = 0
        while (prev_cost - cost) / prev_cost > self.min_improvement:
            iter += 1
            fmt = 'epoch %i[%.2g]: train %s'
            args = (iter,
                    self.f_rate()[0],
                    numpy.mean([self.f_train(*i) for i in train_set], axis=0),
                    )
            if iter % self.validation_frequency == 0:
                metrics = numpy.mean([self.f_eval(*i) for i in valid_set], axis=0)
                fmt += ' valid %s'
                args += (metrics, )
                prev_cost, cost = cost, metrics[0]
            logging.info(fmt, *args)
        return cost


class HF(Trainer):
    '''The hessian free trainer shells out to an external implementation.

    hf.py was implemented by Nicholas Boulanger-Lewandowski and made available
    to the public (yay !). If you don't have a copy of the module handy, this
    class will attempt to download it from github.
    '''

    URL = 'https://raw.github.com/boulanni/theano-hf/master/hf.py'

    def __init__(self, network, **kwargs):
        try:
            import hf
        except:
            # if hf failed to import, try downloading it and saving it locally.
            import os, urllib
            logging.error('hf import failed, attempting to download %s', HF.URL)
            path = os.path.join(os.getcwd(), 'hf.py')
            urllib.urlretrieve(HF.URL, path)
            logging.error('downloaded hf code to %s', path)
            del os
            del urllib
            import hf

        c = [network.J(**kwargs)] + network.monitors
        self.f_eval = theano.function(network.inputs, c)
        self.cg_set = kwargs.pop('cg_set')
        self.opt = hf.hf_optimizer(network.params, network.inputs, network.y, c)

        # fix mapping from kwargs into a dict to send to the hf optimizer
        kwargs['validation_frequency'] = kwargs.pop('validate', sys.maxint)
        for k in set(kwargs) - set(self.opt.train.im_func.func_code.co_varnames[1:]):
            kwargs.pop(k)
        self.kwargs = kwargs

    def train(self, train_set, valid_set=None):
        self.opt.train(train_set, self.cg_set, validation=valid_set, **self.kwargs)


class Cascaded(Trainer):
    '''This trainer uses SGD first, then HF.

    HF is slow, but SGD is inaccurate. So we start with SGD, when HF will be
    making poor updates anyway because the parameters are (probably) not near
    their optimal values. Then, after SGD appears to taper off, we run HF for a
    final tweaking.
    '''

    def __init__(self, network, **kwargs):
        self.network = network

        self.decay = kwargs.pop('decay', 0.9)
        assert self.decay < 1, '--decay must be < 1 for this trainer'

        if kwargs.get('learning_rate') is None:
            kwargs['learning_rate'] = 0.3
        if kwargs.get('min_improvement') is None:
            kwargs['min_improvement'] = 1e-4

        self.kwargs = kwargs

    def train(self, train_set, valid_set=None):
        prev_cost = 1e101
        cost = 1e100
        while (prev_cost - cost) / prev_cost > self.kwargs['min_improvement']:
            logging.info('sgd learning rate: %s', self.kwargs['learning_rate'])
            t = SGD(self.network, **self.kwargs)
            prev_cost, cost = cost, t.train(train_set, valid_set)
            self.kwargs['learning_rate'] *= self.decay
        HF(self.network, **self.kwargs).train(train_set, valid_set)


class FORCE(Trainer):
    '''FORCE is a training method for recurrent nets by Sussillo & Abbott.

    The code here still needs implementation and testing.
    '''

    def __init__(self, network, **kwargs):
        W_in, W_pool, W_out = network.weights

        n = len(W_pool.get_value(shared=True))
        alpha = kwargs.get('learning_rate', 1. / n)
        P = theano.shared(numpy.eye(n).astype(FLOAT) * alpha)

        k = T.dot(P, network.state)
        rPr = 1 + T.dot(network.state, k)
        J = network.J(**kwargs)

        updates = {}
        updates[P] = P - T.dot(k, k) / rPr
        updates[W_pool] = W_pool - J * k / rPr
        updates[W_out] = W_out - J * k / rPr
        updates[b_out] = b_out - alpha * T.grad(J, b_out)

        costs = [J] + network.monitors
        self.f_eval = theano.function(network.inputs, costs)
        self.f_train = theano.function(network.inputs, costs, updates=updates)

    def train(self, train_set, valid_set=None):
        # TODO !
        pass

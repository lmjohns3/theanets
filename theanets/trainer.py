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
import itertools
import numpy as np
import numpy.random as rng
import scipy.optimize
import theano
import theano.tensor as TT
import sys

from . import recurrent

logging = climate.get_logger(__name__)

class Error(Exception): pass  # base class for exceptions in this module.
class PatienceElapsedError(Error): pass
class NoImprovementError(Error): pass


def default_mapper(f, dataset, *args, **kwargs):
    '''Apply a function to each element of a dataset.'''
    return [f(x, *args, **kwargs) for x in dataset]


def ipcluster_mapper(client):
    '''Get a mapper from an IPython.parallel cluster client.'''
    view = client.load_balanced_view()
    def mapper(f, dataset, *args, **kwargs):
        def ff(x):
            return f(x, *args, **kwargs)
        return view.map(ff, dataset).get()
    return mapper


class Trainer(object):
    '''This is a base class for all trainers.'''

    def __init__(self, network, **kwargs):
        super(Trainer, self).__init__()

        self.params = network.params(**kwargs)

        self.J = network.J(**kwargs)
        self.cost_exprs = [self.J]
        self.cost_names = ['J']
        for name, monitor in network.monitors:
            self.cost_names.append(name)
            self.cost_exprs.append(monitor)

        logging.info('compiling evaluation function')
        self.f_eval = theano.function(
            network.inputs, self.cost_exprs, updates=network.updates)

        self.validation_frequency = kwargs.get('validate', 10)
        self.min_improvement = kwargs.get('min_improvement', 0.)
        self.iterations = kwargs.get('num_updates', 1000)
        self.patience = kwargs.get('patience', 100)

        self.shapes = [p.get_value(borrow=True).shape for p in self.params]
        self.counts = [np.prod(s) for s in self.shapes]
        self.starts = np.cumsum([0] + self.counts)[:-1]
        self.dtype = self.params[0].get_value().dtype

        self.best_cost = 1e100
        self.best_iter = 0
        self.best_params = [p.get_value().copy() for p in self.params]

    def flat_to_arrays(self, x):
        x = x.astype(self.dtype)
        return [x[o:o+n].reshape(s) for s, o, n in
                zip(self.shapes, self.starts, self.counts)]

    def arrays_to_flat(self, arrays):
        x = np.zeros((sum(self.counts), ), self.dtype)
        for arr, o, n in zip(arrays, self.starts, self.counts):
            x[o:o+n] = arr.ravel()
        return x

    def set_params(self, targets):
        for param, target in zip(self.params, targets):
            param.set_value(target)

    def evaluate(self, iteration, valid_set):
        costs = np.mean([self.f_eval(*x) for x in valid_set], axis=0)
        improvement = self.best_cost - costs[0] > self.best_cost * self.min_improvement
        marker = ''
        if improvement:
            self.best_cost = costs[0]
            self.best_iter = iteration
            self.best_params = [p.get_value().copy() for p in self.params]
            marker = ' *'
        cost_desc = ' '.join(
            '%s=%.2f' % el for el in zip(self.cost_names, costs))
        logging.info('validation %i %s%s', iteration + 1, cost_desc, marker)
        if iteration - self.best_iter > self.patience:
            raise PatienceElapsedError
        if not improvement:
            raise NoImprovementError

    def train(self, train_set, valid_set=None, **kwargs):
        raise NotImplementedError


class SGD(Trainer):
    '''Stochastic gradient descent network trainer.'''

    def __init__(self, network, **kwargs):
        super(SGD, self).__init__(network, **kwargs)

        self.momentum = kwargs.get('momentum', 0.5)
        self.learning_rate = kwargs.get('learning_rate', 0.01)

        logging.info('compiling %s learning function', self.__class__.__name__)
        self.f_learn = theano.function(
            network.inputs,
            self.cost_exprs,
            updates=list(network.updates) + list(self.learning_updates()))

    def learning_updates(self):
        for param in self.params:
            delta = self.learning_rate * TT.grad(self.J, param)
            if self.momentum > 0:
                velocity = theano.shared(
                    np.zeros_like(param.get_value()), name=param.name + '_vel')
                yield velocity, self.momentum * velocity - delta
                yield param, param + velocity
            else:
                yield param, param - delta

    def train(self, train_set, valid_set=None, **kwargs):
        '''We train over mini-batches and evaluate periodically.'''
        for i in range(self.iterations):
            if not i % self.validation_frequency:
                try:
                    self.evaluate(i, valid_set)
                except KeyboardInterrupt:
                    logging.info('interrupted!')
                    break
                except PatienceElapsedError:
                    logging.info('patience elapsed, bailing out')
                    break
                except NoImprovementError:
                    pass

            try:
                costs = np.mean([self.train_minibatch(*x) for x in train_set], axis=0)
            except KeyboardInterrupt:
                logging.info('interrupted!')
                break

            label = self.__class__.__name__.upper()
            info = ' '.join('%s=%.2f' % el for el in zip(self.cost_names, costs))
            logging.info('%s %i/%i %s', label, i + 1, self.iterations, info)

            yield

        self.set_params(self.best_params)

    def train_minibatch(self, *x):
        return self.f_learn(*x)


class NAG(SGD):
    '''Optimize using Nesterov's Accelerated Gradient (NAG).

    The basic difference between NAG and "classical" momentum in SGD
    optimization approaches is that NAG computes the gradients at the position
    in parameter space where "classical" momentum would put us at the *next*
    step. In symbols, the classical method with momentum m and learning rate a
    updates parameter p by blending the current "velocity" with the current
    gradient:

        v_t+1 = m * v_t - a * grad(p_t)
        p_t+1 = p_t + v_t+1

    while NAG adjusts the update by blending the current "velocity" with the
    next-step gradient (i.e., the gradient at the point where the velocity
    would have taken us):

        v_t+1 = m * v_t - a * grad(p_t + m * v_t)
        p_t+1 = p_t + v_t+1

    The difference here is that the gradient is computed at the place in
    parameter space where we would have stepped using the classical
    technique, in the absence of a new gradient.

    In theory, this helps correct for oversteps during learning: If momentum
    would lead us to overshoot, then the gradient at that overshot place
    will point backwards, toward where we came from. (For details see
    Sutskever, Martens, Dahl, and Hinton, ICML 2013, "On the importance of
    initialization and momentum in deep learning.")
    '''

    def __init__(self, network, **kwargs):
        # due to the way that theano handles updates, we cannot update a
        # parameter twice during the same function call. so, instead of handling
        # everything in the updates for self.f_learn(...), we split the
        # parameter updates into two function calls. the first "prepares" the
        # parameters for the gradient computation by moving the entire model one
        # step according to the current velocity. then the second computes the
        # gradient at that new model position and performs the usual velocity
        # and parameter updates.

        self.params = network.params(**kwargs)

        # set up space for temporary variables used during learning.
        self._steps = []
        self._velocities = []
        for param in self.params:
            v = param.get_value()
            n = param.name
            self._steps.append(theano.shared(np.zeros_like(v), name=n + '_step'))
            self._velocities.append(theano.shared(np.zeros_like(v), name=n + '_vel'))

        # step 1. move to the position in parameter space where we want to
        # compute our gradient.
        prepare = []
        for param, step, velocity in zip(self.params, self._steps, self._velocities):
            prepare.append((step, self.momentum * velocity))
            prepare.append((param, param + step))

        logging.info('compiling NAG adjustment function')
        self.f_prepare = theano.function([], [], updates=prepare)

        super(NAG, self).__init__(network, **kwargs)

    def learning_updates(self):
        # step 2. record the gradient here.
        for param, step, velocity in zip(self.params, self._steps, self._velocities):
            yield velocity, step - self.learning_rate * TT.grad(self.J, param)

        # step 3. update each of the parameters, removing the step that we took
        # to compute the gradient.
        for param, step, velocity in zip(self.params, self._steps, self._velocities):
            yield param, param + velocity - step

    def train_minibatch(self, *x):
        self.f_prepare()
        return self.f_learn(*x)


class RPROP(SGD):
    '''Trainer for neural nets using resilient backpropagation (RPROP).

    The RPROP method uses the same general strategy as SGD, except that only the
    signs of the partial derivatives are taken into account. That is, the step
    size for each parameter is the same regardless of the magnitude of the
    gradient for that parameter.

    The implementation here limits the step size to magnitude 1. (TODO:
    implement variant of RPROP with eta+ and eta- step size adjustments.)
    '''

    def learning_updates(self):
        for param in self.params:
            grad = TT.grad(self.J, param)
            delta = self.learning_rate * ((grad >= 0) - (grad < 0))
            if self.momentum > 0:
                velocity = theano.shared(
                    np.zeros_like(param.get_value()), name=param.name + '_vel')
                yield velocity, self.momentum * velocity - delta
                yield param, param + velocity
            else:
                yield param, param - delta


class Scipy(Trainer):
    '''General trainer for neural nets using `scipy.optimize.minimize`.'''

    METHODS = ('bfgs', 'cg', 'dogleg', 'newton-cg', 'trust-ncg')

    def __init__(self, network, method, **kwargs):
        super(Scipy, self).__init__(network, **kwargs)

        self.method = method

        logging.info('compiling gradient function')
        self.f_grad = theano.function(network.inputs, TT.grad(self.J, self.params))

    def function_at(self, x, train_set):
        self.set_params(self.flat_to_arrays(x))
        costs = np.mean([self.f_eval(*x) for x in train_set], axis=0)
        return costs[0]

    def gradient_at(self, x, train_set):
        self.set_params(self.flat_to_arrays(x))
        grads = [[] for _ in range(len(self.params))]
        for x in train_set:
            for i, g in enumerate(self.f_grad(*x)):
                grads[i].append(np.asarray(g))
        return self.arrays_to_flat([np.mean(g, axis=0) for g in grads])

    def train(self, train_set, valid_set=None, **kwargs):
        def display(x):
            self.set_params(self.flat_to_arrays(x))
            costs = np.mean([self.f_eval(*x) for x in train_set], axis=0)
            cost_desc = ' '.join(
                '%s=%.2f' % el for el in zip(self.cost_names, costs))
            logging.info('scipy %s %i/%i %s',
                         self.method, i + 1, self.iterations, cost_desc)

        for i in range(self.iterations):
            try:
                self.evaluate(i, valid_set)
            except KeyboardInterrupt:
                logging.info('interrupted!')
                break
            except PatienceElapsedError:
                logging.info('patience elapsed, bailing out')
                break
            except NoImprovementError:
                pass

            try:
                res = scipy.optimize.minimize(
                    fun=self.function_at,
                    jac=self.gradient_at,
                    x0=self.arrays_to_flat(self.best_params),
                    args=(train_set, ),
                    method=self.method,
                    callback=display,
                    options=dict(maxiter=self.validation_frequency),
                )
            except KeyboardInterrupt:
                logging.info('interrupted!')
                break

            self.set_params(self.flat_to_arrays(res.x))

            yield

        self.set_params(self.best_params)


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
        import os, tempfile
        try:
            import urllib.request
        except: # Python 2.x
            import urllib
        sys.path.append(tempfile.gettempdir())

        try:
            import hf
        except:
            # if hf failed to import, try downloading it and saving it locally.
            logging.error('hf import failed, attempting to download %s', HF.URL)
            path = os.path.join(tempfile.gettempdir(), 'hf.py')
            try:
                urllib.request.urlretrieve(HF.URL, path)
            except: # Python 2.x
                urllib.urlretrieve(HF.URL, path)
            logging.info('downloaded hf code to %s', path)
            import hf

        self.params = network.params(**kwargs)
        self.opt = hf.hf_optimizer(
            self.params,
            network.inputs,
            network.y,
            [network.J(**kwargs)] + [mon for _, mon in network.monitors],
            network.hiddens[-1] if isinstance(network, recurrent.Network) else None)

        # fix mapping from kwargs into a dict to send to the hf optimizer
        kwargs['validation_frequency'] = kwargs.pop('validate', 1 << 60)
        try:
            func = self.opt.train.__func__.__code__
        except: # Python 2.x
            func = self.opt.train.im_func.func_code
        for k in set(kwargs) - set(func.co_varnames[1:]):
            kwargs.pop(k)
        self.kwargs = kwargs

    def train(self, train_set, valid_set=None, **kwargs):
        self.set_params(self.opt.train(
            train_set, kwargs['cg_set'], validation=valid_set, **self.kwargs))
        yield


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
        S = np.std(pool, axis=0)
        while len(pool) < n:
            x = pool[rng.randint(L)]
            pool.append(x + S * rng.randn(*x.shape))
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

        yield


class Layerwise(Trainer):
    '''This trainer adapts parameters using a variant of layerwise pretraining.

    In this variant, we create "taps" at increasing depths into the original
    network weights, training only those weights that are below the tap. So, for
    a hypothetical binary classifier network with layers [3, 4, 5, 6, 2], we
    would first insert a tap after the first hidden layer (effectively a binary
    classifier in a [3, 4, 2] configuration) and train just that network. Then
    we insert a tap at the next layer (effectively training a [3, 4, 5, 2]
    classifier, re-using the trained weights for the 3x4 layer), and so forth.

    By inserting taps into the original network, we preserve all of the relevant
    settings of noise, dropouts, loss function and the like, in addition to
    obviating the need for copying trained weights around between different
    Network instances.

    I believe this variant of layerwise training was first described by Bengio,
    but I haven't located the citation yet.
    '''

    def __init__(self, network, factory, *args, **kwargs):
        self.network = network
        self.factory = factory
        self.args = args
        self.kwargs = kwargs

    def train(self, train_set, valid_set=None, **kwargs):
        y = self.network.y
        hiddens = list(self.network.hiddens)
        weights = list(self.network.weights)
        biases = list(self.network.biases)

        nout = len(biases[-1].get_value(borrow=True))
        nhids = [len(b.get_value(borrow=True)) for b in biases[:-1]]
        for i in range(1, len(nhids)):
            W, b, _ = self.network._create_layer(nhids[i-1], nout, 'lwout-%d' % i)
            self.network.y = TT.dot(hiddens[i-1], W) + b
            self.network.hiddens = hiddens[:i]
            self.network.weights = weights[:i] + [W]
            self.network.biases = biases[:i] + [b]
            trainer = self.factory(self.network, *self.args, **self.kwargs)
            for _ in trainer.train(train_set, valid_set):
                yield

        # restore the original network configuration and make a final pass to
        # train the last layer.
        self.network.y = y
        self.network.hiddens = hiddens
        self.network.weights = weights
        self.network.biases = biases
        trainer = self.factory(self.network, *self.args, **self.kwargs)
        for _ in trainer.train(train_set, valid_set):
            yield


class FORCE(Trainer):
    '''FORCE is a training method for recurrent nets by Sussillo & Abbott.

    This implementation needs some more love before it will work.
    '''

    def __init__(self, network, **kwargs):
        super(FORCE, Trainer).__init__(network, **kwargs)

    def train(self, train_set, valid_set=None, **kwargs):
        W_in, W_pool, W_out = network.weights

        n = W_pool.get_value(borrow=True).shape[0]
        P = theano.shared(np.eye(n).astype(FLOAT) * self.learning_rate)

        k = TT.dot(P, network.state)
        rPr = TT.dot(network.state, k)
        c = 1. / (1. + rPr)
        dw = network.error(**kwargs) * c * k

        updates = {}
        updates[P] = P - c * TT.outer(k, k)
        updates[W_pool] = W_pool - dw
        updates[W_out] = W_out - dw
        updates[b_out] = b_out - self.learning_rate * TT.grad(J, b_out)

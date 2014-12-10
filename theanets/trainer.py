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

from . import feedforward
from . import recurrent

logging = climate.get_logger(__name__)


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
        '''Evaluate the current model using a validation set.

        Parameters
        ----------
        valid_set : theanets.Dataset
            A set of data to use for evaluating the model. Typically this is
            distinct from the training (and testing) data.

        Returns
        -------
        True iff there was sufficient improvement compared with the last call to
        evaluate.
        '''
        costs = list(zip(
            self.cost_names,
            np.mean([self.f_eval(*x) for x in valid_set], axis=0)))
        marker = ''
        # this is the same as: (J_i - J_f) / J_i > min improvement
        _, J = costs[0]
        if self.best_cost - J > self.best_cost * self.min_improvement:
            self.best_cost = J
            self.best_iter = iteration
            self.best_params = [p.get_value().copy() for p in self.params]
            marker = ' *'
        info = ' '.join('%s=%.2f' % el for el in costs)
        logging.info('validation %i %s%s', iteration + 1, info, marker)
        return iteration - self.best_iter < self.patience

    def train(self, train_set, valid_set=None, **kwargs):
        raise NotImplementedError


class SGD(Trainer):
    '''Stochastic gradient descent network trainer.'''

    def __init__(self, network, **kwargs):
        super(SGD, self).__init__(network, **kwargs)

        self.momentum = kwargs.get('momentum', 0.9)
        self.learning_rate = kwargs.get('learning_rate', 1e-4)

        logging.info('compiling %s learning function', self.__class__.__name__)
        self.f_learn = theano.function(
            network.inputs,
            self.cost_exprs,
            updates=list(network.updates) + list(self.learning_updates()))

    def learning_updates(self):
        for param in self.params:
            delta = self.learning_rate * TT.grad(self.J, param)
            velocity = theano.shared(
                np.zeros_like(param.get_value()), name=param.name + '_vel')
            yield velocity, self.momentum * velocity - delta
            yield param, param + velocity

    def train(self, train_set, valid_set=None, **kwargs):
        '''We train over mini-batches and evaluate periodically.'''
        iteration = 0
        while True:
            if not iteration % self.validation_frequency:
                try:
                    if not self.evaluate(iteration, valid_set):
                        logging.info('patience elapsed, bailing out')
                        break
                except KeyboardInterrupt:
                    logging.info('interrupted!')
                    break

            try:
                costs = list(zip(
                    self.cost_names,
                    np.mean([self.train_minibatch(*x) for x in train_set], axis=0)))
            except KeyboardInterrupt:
                logging.info('interrupted!')
                break

            info = ' '.join('%s=%.2f' % el for el in costs)
            logging.info('%s %i %s', self.__class__.__name__, iteration + 1, info)
            iteration += 1

            yield dict(costs)

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
    would lead us to overshoot, then the gradient at that overshot place will
    point backwards, toward where we came from. (For details see Sutskever,
    Martens, Dahl, and Hinton, ICML 2013, "On the importance of initialization
    and momentum in deep learning." A PDF of this paper is freely available at
    http://jmlr.csail.mit.edu/proceedings/papers/v28/sutskever13.pdf)
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
        self.momentum = kwargs.get('momentum', 0.5)

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


class Rprop(SGD):
    '''Trainer for neural nets using resilient backpropagation.

    The Rprop method uses the same general strategy as SGD (both methods are
    make small parameter adjustments using local derivative information). The
    difference is that in Rprop, only the signs of the partial derivatives are
    taken into account when making parameter updates. That is, the step size for
    each parameter is independent of the magnitude of the gradient for that
    parameter.

    To accomplish this, Rprop maintains a separate learning rate for every
    parameter in the model, and adjusts this learning rate based on the
    consistency of the sign of the gradient of J with respect to that parameter
    over time. Whenever two consecutive gradients for a parameter have the same
    sign, the learning rate for that parameter increases, and whenever the signs
    disagree, the learning rate decreases. This has a similar effect to
    momentum-based SGD methods but effectively maintains parameter-specific
    momentum values.

    The implementation here actually uses the "iRprop-" variant of Rprop
    described in Algorithm 4 from Igel and Huesken (2000), "Improving the Rprop
    Learning Algorithm." This variant resets the running gradient estimates to
    zero in cases where the previous and current gradients have switched signs.
    '''

    def __init__(self, network, **kwargs):
        self.step_increase = kwargs.get('rprop_increase', 1.01)
        self.step_decrease = kwargs.get('rprop_decrease', 0.99)
        self.min_step = kwargs.get('rprop_min_step', 0.)
        self.max_step = kwargs.get('rprop_max_step', 100.)
        super(Rprop, self).__init__(network, **kwargs)

    def learning_updates(self):
        step = self.learning_rate
        self.grads = []
        self.steps = []
        for param in self.params:
            v = param.get_value()
            n = param.name
            self.grads.append(theano.shared(np.zeros_like(v), name=n + '_grad'))
            self.steps.append(theano.shared(np.zeros_like(v) + step, name=n + '_step'))
        for param, step_tm1, grad_tm1 in zip(self.params, self.steps, self.grads):
            grad = TT.grad(self.J, param)
            test = grad * grad_tm1
            same = TT.gt(test, 0)
            diff = TT.lt(test, 0)
            step = TT.minimum(self.max_step, TT.maximum(self.min_step, step_tm1 * (
                TT.eq(test, 0) +
                same * self.step_increase +
                diff * self.step_decrease)))
            grad = grad - diff * grad
            yield param, param - TT.sgn(grad) * step
            yield grad_tm1, grad
            yield step_tm1, step


class RmsProp(SGD):
    '''RmsProp trains neural network models using scaled SGD.

    The Rprop method uses the same general strategy as SGD (both methods are
    make small parameter adjustments using local derivative information). The
    difference here is that as gradients are computed during each parameter
    update, an exponential moving average of squared gradient magnitudes is
    maintained as well. At each update, the EMA is used to compute the
    root-mean-square (RMS) gradient value that's been seen in the recent past.
    The actual gradient is normalized by this RMS scale before being applied to
    update the parameters.

    Like Rprop, this learning method effectively maintains a sort of
    parameter-specific momentum value, but the difference here is that only the
    magnitudes of the gradients are taken into account, rather than the signs.

    The weight parameter for the EMA window is taken from the "momentum" keyword
    argument. If this weight is set to a low value, the EMA will have a short
    memory and will be prone to changing quickly. If the momentum parameter is
    set close to 1, the EMA will have a long history and will change slowly.

    The implementation here is modeled after Graves (2013), "Generating
    Sequences With Recurrent Neural Networks," http://arxiv.org/abs/1308.0850.
    '''

    def __init__(self, network, **kwargs):
        self.clip = kwargs.get('rms_clip', 1000)
        self.ema = kwargs.get('rms_ema', 0.9)
        super(RmsProp, self).__init__(network, **kwargs)

    def learning_updates(self):
        for param in self.params:
            grad = TT.grad(self.J, param).clip(-self.clip, self.clip)
            z = lambda: np.zeros_like(param.get_value())
            g1_ = theano.shared(z(), name=param.name + '_g1')
            g2_ = theano.shared(z(), name=param.name + '_g2')
            vel_ = theano.shared(z(), name=param.name + '_vel')
            g1 = self.ema * g1_ + (1 - self.ema) * grad
            g2 = self.ema * g2_ + (1 - self.ema) * grad * grad
            rms = TT.sqrt(g2 - g1 * g1 + 1e-4)
            vel = self.momentum * vel_ - self.learning_rate * grad / rms
            yield g1_, g1
            yield g2_, g2
            yield vel_, vel
            yield param, param + vel


class Scipy(Trainer):
    '''General trainer for neural nets using `scipy.optimize.minimize`.'''

    METHODS = ('bfgs', 'cg', 'dogleg', 'newton-cg', 'trust-ncg')

    def __init__(self, network, method, **kwargs):
        super(Scipy, self).__init__(network, **kwargs)

        self.method = method
        self.iterations = kwargs.get('num_updates', 100)

        logging.info('compiling gradient function')
        self.f_grad = theano.function(network.inputs, TT.grad(self.J, self.params))

    def function_at(self, x, train_set):
        self.set_params(self.flat_to_arrays(x))
        return np.mean([self.f_eval(*x)[0] for x in train_set])

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
            logging.info('scipy.%s %i %s', self.method, i + 1, cost_desc)

        for i in range(self.iterations):
            try:
                if not self.evaluate(i, valid_set):
                    logging.info('patience elapsed, bailing out')
                    break
            except KeyboardInterrupt:
                logging.info('interrupted!')
                break

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

            yield {'J': res.fun}

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
        yield {'J': -1}


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
        return np.array(pool, dtype=pool[0].dtype)

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
                w.set_value(arr / np.sqrt((arr * arr).sum(axis=1))[:, None])

        # set input (encoding) weights on the network.
        first = lambda x: x[0] if isinstance(x, (tuple, list)) else x
        samples = ifci(first(t) for t in train_set)
        for i, h in enumerate(self.network.hiddens):
            if i == len(self.network.weights):
                break
            w = self.network.weights[i]
            m, k = w.get_value(borrow=True).shape
            arr = np.vstack(Sample.reservoir(samples, k)).T
            logging.info('setting weights for %s: %d x %d <- %s', w.name, m, k, arr.shape)
            w.set_value(arr / np.sqrt((arr * arr).sum(axis=0)))
            samples = ifci(self.network.feed_forward(first(t))[i-1] for t in train_set)

        yield {'J': -1}


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
    removing the need for copying trained weights around between different
    Network instances.
    '''

    def __init__(self, network, factory, *args, **kwargs):
        self.network = network
        self.factory = factory
        self.args = args
        self.kwargs = kwargs

    def train(self, train_set, valid_set=None, **kwargs):
        '''Train a network using a layerwise strategy.

        Parameters
        ----------
        train_set : :class:`theanets.Dataset`
            A training set to use while training the weights in our network.
        valid_set : :class:`theanets.Dataset`
            A validation set to use while training the weights in our network.

        Returns
        -------
        Generates a series of cost values as the network weights are tuned.
        '''
        net = self.network

        y = net.y
        hiddens = list(net.hiddens)
        weights = list(net.weights)
        biases = list(net.biases)

        nout = len(biases[-1].get_value(borrow=True))
        nhids = [len(b.get_value(borrow=True)) for b in biases]
        for i in range(1, len(weights) + 1 if net.tied_weights else len(nhids)):
            net.hiddens = hiddens[:i]
            if net.tied_weights:
                net.weights = [weights[i-1]]
                net.biases = [biases[i-1]]
                for j in range(i - 1, -1, -1):
                    net.hiddens.append(TT.dot(net.hiddens[-1], weights[j].T))
                net.y = net._output_func(net.hiddens.pop())
            else:
                W, _ = net.create_weights(nhids[i-1], nout, 'layerwise')
                b, _ = net.create_bias(nout, 'layerwise')
                net.weights = [weights[i-1], W]
                net.biases = [biases[i-1], b]
                net.y = net._output_func(TT.dot(hiddens[i-1], W) + b)
            logging.info('layerwise: training weights %s', net.weights[0].name)
            trainer = self.factory(net, *self.args, **self.kwargs)
            for costs in trainer.train(train_set, valid_set):
                yield costs

        net.y = y
        net.hiddens = hiddens
        net.weights = weights
        net.biases = biases


class UnsupervisedPretrainer(Trainer):
    '''Train a discriminative model using an unsupervised pre-training step.

    This trainer is a bit of glue code that creates a "shadow" autoencoder based
    on a current network model, trains the autoencoder, and then transfers the
    trained weights back to the original model.

    This code is intended mostly as a proof-of-concept; more elaborate training
    strategies are certainly possible but should be coded outside the core
    package.
    '''

    def __init__(self, network, *args, **kwargs):
        self.network = network
        self.args = args
        self.kwargs = kwargs

    def train(self, train_set, valid_set=None, **kwargs):
        # construct a copy of the input network, with tied weights in an
        # autoencoder configuration.
        layers = self.network.layers[:-1]
        ae = feedforward.Autoencoder(
            tied_weights=True,
            layers=layers[:-1] + layers[::-1],
            hidden_activation=self.network.hidden_activation,
            output_activation='linear')

        # copy the current weights into the autoencoder.
        for i in range(len(layers) - 1):
            ae.weights[i].set_value(self.network.get_weights(i))
            ae.biases[i].set_value(self.network.get_biases(i))

        # train the autoencoder using a layerwise strategy.
        pre = Layerwise(ae, *self.args, **self.kwargs)
        for costs in pre.train(train_set, valid_set=valid_set, **kwargs):
            yield costs

        # copy the trained autoencoder weights into our original model.
        for i in range(len(layers) - 1):
            self.network.weights[i].set_value(ae.get_weights(i))
            self.network.biases[i].set_value(ae.get_biases(i))

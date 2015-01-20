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

'''This module contains optimization methods for neural networks.

Several of the optimization methods --- namely, :class:`SGD`, :class:`Rprop`,
:class:`RmsProp`, :class:`ADADELTA`, :class:`HF`, and :class:`Scipy` --- are
general-purpose optimization routines that happen to be pretty good for training
neural networks. Other methods --- :class:`Sample`,
:class:`SupervisedPretrainer`, and :class:`UnsupervisedPretrainer` --- are
specific to neural networks. Despite the difference in generality, all of the
training routines implemented here assume that a :class:`Network
<theanets.feedforward.Network>` is being optimized.

Most of the general-purpose optimization routines in this module are based on
the :class:`SGD` parent and optimize the cost function at hand by taking small
steps in the general direction of the local gradient of the cost function. Such
stochastic gradient optimization techniques are not bad, in the sense that they
will generally always take steps that lower the cost function, but because they
use local gradient information, they are not guaranteed to find a global optimum
for nonlinear cost functions. Whether this is a problem or not depends on your
task, but these approaches have been shown to be quite useful in the past couple
decades of machine learning research.
'''

import climate
import itertools
import numpy as np
import numpy.random as rng
import scipy.optimize
import theano
import theano.tensor as TT
import sys

from . import feedforward
from . import layers

logging = climate.get_logger(__name__)


def default_mapper(f, dataset, *args, **kwargs):
    '''Apply (map) a function to each element of a dataset.'''
    return [f(x, *args, **kwargs) for x in dataset]


def ipcluster_mapper(client):
    '''Get a mapper from an IPython.parallel cluster client.

    This helper is experimental and not currently used.

    Parameters
    ----------
    client : :ipy:`IPython.parallel.Client`
        A client for an IPython cluster. The dataset will be processed by
        distributing it across the cluster.

    Returns
    -------
    mapper : callable
        A callable that can be used to map a dataset to a function across an
        IPython cluster.
    '''
    view = client.load_balanced_view()
    def mapper(f, dataset, *args, **kwargs):
        def ff(x):
            return f(x, *args, **kwargs)
        return view.map(ff, dataset).get()
    return mapper


class Trainer(object):
    '''All trainers derive from this base class.'''

    def __init__(self, network, **kwargs):
        super(Trainer, self).__init__()

        self.validation_frequency = kwargs.get('validate', 10)
        self.min_improvement = kwargs.get('min_improvement', 0.)
        self.patience = kwargs.get('patience', 100)

        self.params = network.params(**kwargs)
        self._shapes = [p.get_value(borrow=True).shape for p in self.params]
        self._counts = [np.prod(s) for s in self._shapes]
        self._starts = np.cumsum([0] + self._counts)[:-1]
        self._dtype = self.params[0].get_value().dtype

        self._best_cost = 1e100
        self._best_iter = 0
        self._best_params = [p.get_value().copy() for p in self.params]

        self.J = network.J(**kwargs)
        self._monitor_exprs = [self.J]
        self._monitor_names = ['J']
        for name, monitor in network.monitors:
            self._monitor_names.append(name)
            self._monitor_exprs.append(monitor)

        logging.info('compiling evaluation function')
        self.f_eval = theano.function(
            network.inputs, self._monitor_exprs, updates=network.updates)

    def flat_to_arrays(self, x):
        x = x.astype(self._dtype)
        return [x[o:o+n].reshape(s) for s, o, n in
                zip(self._shapes, self._starts, self._counts)]

    def arrays_to_flat(self, arrays):
        x = np.zeros((sum(self._counts), ), self._dtype)
        for arr, o, n in zip(arrays, self._starts, self._counts):
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
            self._monitor_names,
            np.mean([self.f_eval(*x) for x in valid_set], axis=0)))
        marker = ''
        # this is the same as: (J_i - J_f) / J_i > min improvement
        _, J = costs[0]
        if self._best_cost - J > self._best_cost * self.min_improvement:
            self._best_cost = J
            self._best_iter = iteration
            self._best_params = [p.get_value().copy() for p in self.params]
            marker = ' *'
        info = ' '.join('%s=%.2f' % el for el in costs)
        logging.info('validation %i %s%s', iteration + 1, info, marker)
        return iteration - self._best_iter < self.patience

    def train(self, train_set, valid_set=None, **kwargs):
        raise NotImplementedError


class SGD(Trainer):
    '''Stochastic gradient descent network trainer.

    '''

    def __init__(self, network, **kwargs):
        super(SGD, self).__init__(network, **kwargs)

        self.clip = kwargs.get('gradient_clip', 1e6)
        self.max_norm = kwargs.get('max_gradient_norm', 1e6)
        self.momentum = kwargs.get('momentum', 0.9)
        self.learning_rate = kwargs.get('learning_rate', 1e-4)

        logging.info('compiling %s learning function', self.__class__.__name__)
        self.f_learn = theano.function(
            network.inputs,
            updates=list(network.updates) + list(self.learning_updates()))

    def learning_updates(self):
        for param, grad in zip(self.params, self.clipped_gradients()):
            vel_tm1 = self.shared_like(param, 'vel')
            vel_t = self.momentum * vel_tm1 - self.learning_rate * grad
            yield vel_tm1, vel_t
            yield param, param + vel_t

    def clipped_gradients(self, params=None):
        for grad in TT.grad(self.J, params or self.params):
            clip = TT.clip(grad, -self.clip, self.clip)
            norm = TT.sqrt((grad * grad).sum())
            yield clip * TT.minimum(1, self.max_norm / norm)

    @staticmethod
    def shared_like(param, name, init=0):
        return theano.shared(np.zeros_like(param.get_value()) + init,
                             name='{}_{}'.format(param.name, name))

    def train(self, train_set, valid_set=None, **kwargs):
        '''We compute gradients using mini-batches and evaluate periodically.

        This trainer encompasses an important subset of optimization algorithms
        that use local gradient information to make iterative adjustments to
        minimize a loss function.

        Parameters
        ----------
        train_set : :class:`theanets.dataset.Dataset`
            A training set to use while training the weights in our network.
        valid_set : :class:`theanets.dataset.Dataset`
            A validation set to use while training the weights in our network.

        Returns
        -------
        Generates a series of cost values as the network weights are tuned.

        '''
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
                [self.train_minibatch(*x) for x in train_set]
            except KeyboardInterrupt:
                logging.info('interrupted!')
                break

            try:
                costs = list(zip(
                    self._monitor_names,
                    np.mean([self.f_eval(*x) for i, x in zip(range(3), train_set)], axis=0)))
            except KeyboardInterrupt:
                logging.info('interrupted!')
                break

            info = ' '.join('%s=%.2f' % el for el in costs)
            logging.info('%s %i %s', self.__class__.__name__, iteration + 1, info)
            iteration += 1

            yield dict(costs)

        self.set_params(self._best_params)

    def train_minibatch(self, *x):
        self.f_learn(*x)


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
        self._vels = []
        for param in self.params:
            self._steps.append(self.shared_like(param, 'step'))
            self._vels.append(self.shared_like(param, 'vel'))

        # step 1. move to the position in parameter space where we want to
        # compute our gradient.
        prepare = []
        for param, step, vel in zip(self.params, self._steps, self._vels):
            prepare.append((step, self.momentum * vel))
            prepare.append((param, param + step))

        logging.info('compiling NAG pre-step function')
        self.f_prepare = theano.function([], [], updates=prepare)

        super(NAG, self).__init__(network, **kwargs)

    def learning_updates(self):
        # step 2. record the gradient here.
        for grad, step, vel in zip(self.clipped_gradients(), self._steps, self._vels):
            yield vel, step - self.learning_rate * grad

        # step 3. update each of the parameters, removing the step that we took
        # to compute the gradient.
        for param, step, vel in zip(self.params, self._steps, self._vels):
            yield param, param + vel - step

    def train_minibatch(self, *x):
        self.f_prepare()
        self.f_learn(*x)


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
        for param, grad in zip(self.params, self.clipped_gradients()):
            grad_tm1 = self.shared_like(param, 'grad')
            step_tm1 = self.shared_like(param, 'step', self.learning_rate)
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

    The RmsProp method uses the same general strategy as SGD (both methods are
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
        self.ewma = float(np.exp(-np.log(2) / kwargs.get('rms_halflife', 7)))
        super(RmsProp, self).__init__(network, **kwargs)

    def learning_updates(self):
        for param, grad in zip(self.params, self.clipped_gradients()):
            g1_tm1 = self.shared_like(param, 'g1_ewma')
            g2_tm1 = self.shared_like(param, 'g2_ewma')
            vel_tm1 = self.shared_like(param, 'vel')
            g1_t = self.ewma * g1_tm1 + (1 - self.ewma) * grad
            g2_t = self.ewma * g2_tm1 + (1 - self.ewma) * grad * grad
            rms = TT.sqrt(g2_t - g1_t * g1_t + 1e-4)
            vel_t = self.momentum * vel_tm1 - grad * self.learning_rate / rms
            yield g1_tm1, g1_t
            yield g2_tm1, g2_t
            yield vel_tm1, vel_t
            yield param, param + vel_t


class ADADELTA(RmsProp):
    '''ADADELTA trains neural network models using scaled SGD.

    The ADADELTA method uses the same general strategy as SGD (both methods are
    make small parameter adjustments using local derivative information). The
    difference here is that as gradients are computed during each parameter
    update, an exponential weighted moving average gradient value, as well as an
    exponential weighted moving average of recent parameter steps, are
    maintained as well. The actual gradient is normalized by the ratio of the
    parameter step RMS values to the gradient RMS values.

    Like Rprop and RmsProp, this learning method effectively maintains a sort of
    parameter-specific momentum value. The primary difference between this
    method and RmsProp is that ADADELTA additionally incorporates a sliding
    window of RMS parameter steps.

    The implementation here is modeled after Zeiler (2012), "ADADELTA: An
    adaptive learning rate method," available at http://arxiv.org/abs/1212.5701.
    '''

    def learning_updates(self):
        eps = self.learning_rate
        for param, grad in zip(self.params, self.clipped_gradients()):
            x2_tm1 = self.shared_like(param, 'x2_ewma')
            g2_tm1 = self.shared_like(param, 'g2_ewma')
            g2_t = self.ewma * g2_tm1 + (1 - self.ewma) * grad * grad
            delta = grad * TT.sqrt(x2_tm1 + eps) / TT.sqrt(g2_t + eps)
            x2_t = self.ewma * x2_tm1 + (1 - self.ewma) * delta * delta
            yield g2_tm1, g2_t
            yield x2_tm1, x2_t
            yield param, param - delta


class Scipy(Trainer):
    '''General trainer for neural nets using ``scipy``.

    This trainer class shells out to :func:`scipy.optimize.minimize` to minize
    the network loss. All network operations are carried out using the
    computation graph implemented in Theano, while all optimization procedures
    are carried out using Scipy's code.

    This separation can limit the speedup you might experience while optimizing
    your network's loss, because computations are only carried out on the GPU
    within the Theano graph; any results are passed across the PCI bus back into
    main memory so that the Scipy code can process them.

    The specific algorithms available in this trainer are given in the `METHODS`
    list.
    '''

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
                '%s=%.2f' % el for el in zip(self._monitor_names, costs))
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
                    x0=self.arrays_to_flat(self._best_params),
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

        self.set_params(self._best_params)


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
            network.outputs[0],
            [network.J(**kwargs)] + [mon for _, mon in network.monitors],
            None)

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
        train_set : :class:`theanets.dataset.Dataset`
            A training set to use while training the weights in our network.
        valid_set : :class:`theanets.dataset.Dataset`
            A validation set to use while training the weights in our network.

        Returns
        -------
        Generates a series of cost values as the network weights are tuned.
        '''
        net = self.network
        outact = net.output_activation
        tied = getattr(net, 'tied_weights', False)
        original = list(net.layers)
        L = len(original)
        if tied:
            L //= 2
        else:
            L -= 1
        def addl(*args, **kwargs):
            l = layers.build(*args, **kwargs)
            l.reset()
            net.layers.append(l)
        for i in range(1, L - 1):
            logging.info('layerwise: training %s', original[i].name)
            net.layers = original[:i+1]
            if tied:
                for j in range(i, 1, -1):
                    addl('tied', partner=original[j], name='lw{}'.format(j))
                addl('tied', partner=original[1], name='out', activation=outact)
            else:
                addl('feedforward',
                     name='lwout',
                     nin=original[i].nout,
                     nout=original[-1].nout,
                     activation=outact)
            trainer = self.factory(net, *self.args, **self.kwargs)
            for costs in trainer.train(train_set, valid_set):
                yield costs
        net.layers = original


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
        lls = self.network.layers[:-1]
        for l in lls[::-1][:-1]:
            lls.append(layers.build('tied', partner=l, activation=self.network.hidden_activation))
        lls.append(layers.build('tied', partner=lls[0], activation='linear'))
        ae = feedforward.Autoencoder(tied_weights=True, layers=lls)

        # copy the current weights into the autoencoder.
        for i in range(len(self.network.layers) - 1):
            ae.get_weights(i).set_value(self.network.get_weights(i).get_value())
            ae.get_biases(i).set_value(self.network.get_biases(i).get_value())

        # train the autoencoder using a layerwise strategy.
        pre = Layerwise(ae, *self.args, **self.kwargs)
        for costs in pre.train(train_set, valid_set=valid_set, **kwargs):
            yield costs

        # copy the trained autoencoder weights into our original model.
        for i in range(len(self.network.layers) - 1):
            self.network.get_weights(i).set_value(ae.get_weights(i).get_value())
            self.network.get_biases(i).set_value(ae.get_biases(i).get_value())

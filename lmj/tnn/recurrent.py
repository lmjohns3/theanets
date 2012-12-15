import numpy
import numpy.random as rng
import theano
import theano.tensor as T

import ff

FLOAT = ff.FLOAT


class Network(ff.Network):
    def __init__(self, input, size, output, nonlinearity=None, gain=1.2, damping=0.01):
        if nonlinearity is None:
            nonlinearity = T.nnet.sigmoid
        self.nonlinearity = nonlinearity

        self.x = T.matrix('x')
        self.s = theano.shared(numpy.zeros(size, dtype=FLOAT), name='state')

        arr = rng.normal(size=(input, size)) / numpy.sqrt(input + size)
        W_in = theano.shared(arr.astype(FLOAT), name='W_in')

        arr = rng.normal(size=(size, size)) * gain / numpy.sqrt(size + size)
        W_pool = theano.shared(arr.astype(FLOAT), name='W_pool')

        arr = rng.normal(size=(size, output)) / numpy.sqrt(size + output)
        W_out = theano.shared(arr.astype(FLOAT), name='W_out')
        b_out = theano.shared(numpy.zeros((output, ), FLOAT), name='b_out')

        self.hiddens = [self.s]
        self.weights = [W_in, W_pool, W_out]
        self.biases = [b_out]

        logging.info('%d total network parameters', (input + size + output) * (size + 1))

        st = nonlinearity(T.dot(self.x, W_in) + T.dot(self.s, W_pool) + b_pool)
        self.st = damping * s + (1 - damping) * st
        self.y = T.dot(self.st, W_out) + b_out

        self.f = theano.function(*self.args, updates={self.s: self.st})

    @property
    def inputs(self):
        return [self.x]

    @property
    def args(self):
        return [self.x], [self.y]


class FORCE(ff.Trainer):
    def __init__(self, network, **kwargs):
        W_in, W_pool, W_out = network.weights

        n = len(W_pool.get_value(shared=True))
        alpha = kwargs.get('learning_rate', 1. / n)
        P = theano.shared(numpy.eye(n).astype(FLOAT) * alpha)

        k = T.dot(P, network.s)
        rPr = 1 + T.dot(network.s, k)
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
        pass

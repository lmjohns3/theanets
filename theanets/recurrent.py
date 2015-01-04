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

'''This file contains recurrent network structures.'''

import climate
import numpy as np
import numpy.random as rng
import theano
import theano.tensor as TT

#from theano.tensor.shared_randomstreams import RandomStreams
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

from . import feedforward as ff

logging = climate.get_logger(__name__)


def batches(samples, labels=None, steps=100, batch_size=64):
    '''Return a callable that generates samples from a dataset.

    Parameters
    ----------
    samples : ndarray (time-steps, data-dimensions)
        An array of data. Rows in this array correspond to time steps, and
        columns to variables.
    labels : ndarray (time-steps, label-dimensions), optional
        An array of data. Rows in this array correspond to time steps, and
        columns to labels.
    steps : int, optional
        Generate samples of this many time steps. Defaults to 100.
    batch_size : int, optional
        Generate this many samples per call. Defaults to 64. This must match the
        batch_size parameter that was used when creating the recurrent network
        that will process the data.

    Returns
    -------
    callable :
        A callable that can be used inside a dataset for training a recurrent
        network.
    '''
    def unlabeled_sample():
        xs = np.zeros((steps, batch_size, samples.shape[1]), ff.FLOAT)
        for i in range(batch_size):
            j = rng.randint(len(samples) - steps)
            xs[:, i, :] = samples[j:j+steps]
        return [xs]
    def labeled_sample():
        xs = np.zeros((steps, batch_size, samples.shape[1]), ff.FLOAT)
        ys = np.zeros((steps, batch_size, labels.shape[1]), ff.FLOAT)
        for i in range(batch_size):
            j = rng.randint(len(samples) - steps)
            xs[:, i, :] = samples[j:j+steps]
            ys[:, i, :] = labels[j:j+steps]
        return [xs, ys]
    return unlabeled_sample if labels is None else labeled_sample


class Network(ff.Network):
    '''A fully connected recurrent network with one input and one output layer.

    Parameters
    ----------
    layers : sequence of int
        A sequence of integers specifying the number of units at each layer. As
        an example, layers=(10, 20, 3) has one "input" layer with 10 units, one
        "hidden" layer with 20 units, and one "output" layer with 3 units. That
        is, inputs should be of length 10, and outputs will be of length 3.

    recurrent_layers : sequence of int, optional
        A sequence of integers specifying the indices of recurrent layers in the
        network. Non-recurrent network layers receive input only from the
        preceding layers for a given input, while recurrent layers also receive
        input from the output of the recurrent layer from the previous time
        step. Defaults to [len(layers) // 2 - 1], i.e., the "middle" layer of
        the network is the only recurrent layer.

    recurrent_sparsity : float in (0, 1), optional
        Ensure that the given fraction of recurrent model weights is initialized
        to zero. Defaults to 0, which makes all recurrent weights nonzero.

    hidden_activation : str, optional
        The name of an activation function to use on hidden network units.
        Defaults to 'sigmoid'.

    output_activation : str, optional
        The name of an activation function to use on output units. Defaults to
        'linear'.

    rng : theano.RandomStreams, optional
        Use a specific Theano random number generator. A new one will be created
        if this is None.

    input_noise : float, optional
        Standard deviation of desired noise to inject into input.

    hidden_noise : float, optional
        Standard deviation of desired noise to inject into hidden unit
        activation output.

    input_dropouts : float, optional
        Proportion of input units to randomly set to 0.

    hidden_dropouts : float, optional
        Proportion of hidden unit activations to randomly set to 0.

    recurrent_error_start : int, optional
        Compute error metrics starting at this time step. (Defaults to 3.)
    '''

    @property
    def error_start(self):
        return self.kwargs.get('recurrent_error_start', 3)

    def setup_vars(self):
        '''Setup Theano variables for our network.

        Returns
        -------
        vars : list of theano variables
            A list of the variables that this network requires as inputs.
        '''
        # the first dimension indexes time, the second indexes the elements of
        # each minibatch, and the third indexes the variables in a given frame.
        self.x = TT.tensor3('x')

        # we store the initial state of recurrent hidden units in shared
        # variables indexed by this dictionary.
        self.h0 = {}

        return [self.x]

    def setup_layers(self, **kwargs):
        '''
        '''
        count = 0

        noise = kwargs.get('input_noise', 0)
        dropout = kwargs.get('input_dropouts', 0)
        x = self._add_noise(self.x, noise, dropout)

        kw = dict(
            sparse=kwargs.get('recurrent_sparsity', 0),
            radius=kwargs.get('recurrent_radius', 0),
            noise=kwargs.get('hidden_noise', 0),
            dropout=kwargs.get('hidden_dropouts', 0),
            factors=kwargs.get('mrnn_factors', 0),
        )
        layers = kwargs.get('layers')
        recurrent = set(kwargs.get('recurrent_layers', [len(layers) // 2 - 1]))
        for i, (nin, nout) in enumerate(zip(layers[:-1], layers[1:])):
            z = self.hiddens and self.hiddens[-1] or x
            add = self.add_feedforward_layer
            if i in recurrent:
                add = self.add_recurrent_layer
            count += add(z, nin, nout, label=i, **kw)

        self.hiddens.pop()
        self.y = self._output_func(self.preacts[-1])
        logging.info('%d total network parameters', count)


class Autoencoder(Network):
    '''An autoencoder network attempts to reproduce its input.
    '''

    @property
    def cost(self):
        err = self.y - self.x
        return TT.mean((err * err).sum(axis=2)[self.error_start:])


class Predictor(Autoencoder):
    '''A predictor network attempts to predict its next time step.
    '''

    @property
    def cost(self):
        # we want the network to predict the next time step. y is the prediction
        # (output of the network), so we want y[0] to match x[1], y[1] to match
        # x[2], and so forth.
        err = self.x[1:] - self.generate_prediction(self.y)[:-1]
        return TT.mean((err * err).sum(axis=2)[self.error_start:])

    def generate_prediction(self, y):
        '''Given outputs from each time step, map them to subsequent inputs.

        This defaults to the identity transform, i.e., the output from one time
        step is treated as the input to the next time step with no
        transformation. Override this method in a subclass to provide, e.g.,
        predictions based on random samples, lookups in a dictionary, etc.

        Parameters
        ----------
        y : theano variable
            A symbolic variable representing the "raw" output of the recurrent
            predictor.

        Returns
        -------
        y : theano variable
            A symbolic variable representing the inputs for the next time step.
        '''
        return y


class Regressor(Network):
    '''A regressor attempts to produce a target output.'''

    def setup_vars(self):
        '''Setup Theano variables for our network.

        Returns
        -------
        vars : list of theano variables
            A list of the variables that this network requires as inputs.
        '''
        super(Regressor, self).setup_vars()

        # for a regressor, k specifies the correct outputs for a given input.
        self.k = TT.tensor3('k')

        return [self.x, self.k]

    @property
    def cost(self):
        err = self.y - self.k
        return TT.mean((err * err).sum(axis=2)[self.error_start:])


class Classifier(Network):
    '''A classifier attempts to match a 1-hot target output.'''

    def __init__(self, **kwargs):
        kwargs['output_activation'] = 'softmax'

        super(Classifier, self).__init__(**kwargs)

    def setup_vars(self):
        '''Setup Theano variables for our network.

        Returns
        -------
        vars : list of theano variables
            A list of the variables that this network requires as inputs.
        '''
        super(Classifier, self).setup_vars()

        # for a classifier, k specifies the correct labels for a given input.
        self.k = TT.ivector('k')

        return [self.x, self.k]

    @property
    def cost(self):
        return -TT.mean(TT.log(self.y)[TT.arange(self.k.shape[0]), self.k])

    @property
    def accuracy(self):
        '''Compute the percent correct classifications.'''
        return 100 * TT.mean(TT.eq(TT.argmax(self.y, axis=1), self.k))

    @property
    def monitors(self):
        yield 'acc', self.accuracy
        for i, h in enumerate(self.hiddens):
            yield 'h{}<0.1'.format(i+1), 100 * (abs(h) < 0.1).mean()
            yield 'h{}<0.9'.format(i+1), 100 * (abs(h) < 0.9).mean()

    def classify(self, x):
        return self.predict(x).argmax(axis=1)

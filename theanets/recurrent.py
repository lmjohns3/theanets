'''This module contains recurrent network structures.'''

import climate
import numpy as np
import numpy.random as rng
import theano
import theano.tensor as TT

from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

from . import feedforward
from . import layers

logging = climate.get_logger(__name__)

FLOAT = theano.config.floatX


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
        xs = np.zeros((steps, batch_size, samples.shape[1]), FLOAT)
        for i in range(batch_size):
            j = rng.randint(len(samples) - steps)
            xs[:, i, :] = samples[j:j+steps]
        return [xs]
    def labeled_sample():
        xs = np.zeros((steps, batch_size, samples.shape[1]), FLOAT)
        ys = np.zeros((steps, batch_size, labels.shape[1]), FLOAT)
        for i in range(batch_size):
            j = rng.randint(len(samples) - steps)
            xs[:, i, :] = samples[j:j+steps]
            ys[:, i, :] = labels[j:j+steps]
        return [xs, ys]
    return unlabeled_sample if labels is None else labeled_sample


class Network(feedforward.Network):
    '''A fully connected recurrent network with one input and one output layer.

    Parameters
    ----------
    layers : sequence of int, tuple, dict, or :class:`Layer <layers.Layer>`
        A sequence of values specifying the layer configuration for the network.
        For more information, please see :ref:`creating-specifying-layers`.
    hidden_activation : str, optional
        The name of an activation function to use on hidden network layers by
        default. Defaults to 'logistic'.
    output_activation : str, optional
        The name of an activation function to use on the output layer by
        default. Defaults to 'linear'.
    rng : theano RandomStreams object, optional
        Use a specific Theano random number generator. A new one will be created
        if this is None.
    input_noise : float, optional
        Standard deviation of desired noise to inject into input.
    hidden_noise : float, optional
        Standard deviation of desired noise to inject into hidden unit
        activation output.
    input_dropouts : float in [0, 1], optional
        Proportion of input units to randomly set to 0.
    hidden_dropouts : float in [0, 1], optional
        Proportion of hidden unit activations to randomly set to 0.
    decode_from : positive int, optional
        Any of the hidden layers can be tapped at the output. Just specify a
        value greater than 1 to tap the last N hidden layers. The default is 1,
        which decodes from just the last layer.

    Attributes
    ----------
    layers : list of :class:`theanets.Layer`
        A list of the layers in this network model.
    kwargs : dict
        A dictionary containing the keyword arguments used to construct the
        network.
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

        return [self.x]


class Autoencoder(Network, feedforward.Autoencoder):
    '''An autoencoder network attempts to reproduce its input.
    '''


class Predictor(Autoencoder):
    '''A predictor network attempts to predict its next time step.
    '''

    @property
    def error(self):
        # we want the network to predict the next time step. if y =
        # self.outputs[-1] is output of the network and f(y) gives the
        # prediction, then we want f(y)[0] to match x[1], f(y)[1] to match x[2],
        # and so forth.
        error = self.x[1:] - self.generate_prediction(self.outputs[-1])[:-1]
        err = error[self.error_start:]
        return TT.mean((err * err).sum(axis=-1))

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


class Regressor(Network, feedforward.Regressor):
    '''A regressor attempts to produce a target output.'''

    def setup_vars(self):
        '''Setup Theano variables for our network.

        Returns
        -------
        vars : list of theano variables
            A list of the variables that this network requires as inputs.
        '''
        super(Regressor, self).setup_vars()

        # for a regressor, this specifies the correct outputs for a given input.
        self.targets = TT.tensor3('targets')

        return [self.x, self.targets]

    @property
    def error(self):
        err = (self.outputs[-1] - self.targets)[self.error_start:]
        return TT.mean((err * err).sum(axis=-1))


class Classifier(Network, feedforward.Classifier):
    '''A classifier attempts to match a 1-hot target output.'''

    def setup_vars(self):
        '''Setup Theano variables for our network.

        Returns
        -------
        vars : list of theano variables
            A list of the variables that this network requires as inputs.
        '''
        super(Classifier, self).setup_vars()

        # for a classifier, this specifies the correct labels for a given input.
        self.labels = TT.imatrix('labels')

        return [self.x, self.labels]

    @property
    def error(self):
        '''Returns a theano computation of cross entropy.'''
        out = self.outputs[-1]
        # flatten all but last components of the output and labels
        count = (out.shape[0] - self.error_start) * out.shape[1]
        correct = TT.reshape(self.labels[self.error_start:], (count, ))
        prob = TT.reshape(out[self.error_start:], (count, out.shape[2]))
        return -TT.mean(TT.log(prob[TT.arange(count), correct]))

    @property
    def accuracy(self):
        '''Returns a theano computation of percent correct classifications.'''
        out = self.outputs[-1]
        predict = TT.argmax(out, axis=-1)
        correct = TT.eq(predict, self.labels)
        return TT.cast(100, FLOAT) * TT.mean(correct.flatten())

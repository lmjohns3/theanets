# -*- coding: utf-8 -*-

r'''Feedforward layers for neural network computation graphs.'''

from __future__ import division

import climate
import numpy as np
import theano
import theano.sparse as SS
import theano.tensor as TT

from .base import Layer, random_matrix

logging = climate.get_logger(__name__)

__all__ = [
    'Classifier',
    'Feedforward',
    'Input',
    'Maxout',
    'Tied',
]


class Input(Layer):
    '''The input of a network is a special type of layer with no parameters.

    Input layers essentially add only noise to the input data (if desired), but
    otherwise reproduce their inputs exactly.
    '''

    def __init__(self, **kwargs):
        kwargs['inputs'] = 0
        kwargs['activation'] = 'linear'
        super(Input, self).__init__(**kwargs)

    def to_spec(self):
        '''Create a specification for this layer.

        Returns
        -------
        spec : int
            A single integer specifying the size of this layer.
        '''
        return self.size


class Feedforward(Layer):
    '''A feedforward neural network layer performs a transform of its input.

    More precisely, feedforward layers as implemented here perform an affine
    transformation of their input, followed by a potentially nonlinear
    "activation" function performed elementwise on the transformed input.

    Feedforward layers are the fundamental building block on which most neural
    network models are built.
    '''

    def transform(self, inputs):
        '''Transform the inputs for this layer into an output for the layer.

        Parameters
        ----------
        inputs : dict of Theano expressions
            Symbolic inputs to this layer, given as a dictionary mapping string
            names to Theano expressions. See :func:`Layer.connect`.

        Returns
        -------
        outputs : dict of Theano expressions
            A map from string output names to Theano expressions for the outputs
            from this layer. This layer type generates a "pre" output that gives
            the unit activity before applying the layer's activation function,
            and an "out" output that gives the post-activation output.
        updates : list of update pairs
            An empty list of updates to apply from this layer.
        '''
        def _dot(x, y):
            if isinstance(x, SS.SparseVariable):
                return SS.structured_dot(x, y)
            else:
                return TT.dot(x, y)
        def weight(n):
            return 'w' if len(self.inputs) == 1 else 'w_{}'.format(n)
        xws = ((inputs[n], self.find(weight(n))) for n in self.inputs)
        pre = sum(_dot(x, w) for x, w in xws) + self.find('b')
        return dict(pre=pre, out=self.activate(pre)), []

    def setup(self):
        '''Set up the parameters and initial values for this layer.'''
        for name, size in self.inputs.items():
            label = 'w' if len(self.inputs) == 1 else 'w_{}'.format(name)
            self.add_weights(label, size, self.size)
        self.add_bias('b', self.size)


class Classifier(Feedforward):
    '''A classifier layer performs a softmax over a linear input transform.

    Classifier layers are typically the "output" layer of a classifier network.

    This layer type really only wraps the output activation of a standard
    :class:`Feedforward` layer.
    '''

    def __init__(self, **kwargs):
        kwargs['activation'] = 'softmax'
        super(Classifier, self).__init__(**kwargs)


class Tied(Layer):
    '''A tied-weights feedforward layer shadows weights from another layer.

    Tied weights are typically featured in some types of autoencoder models
    (e.g., PCA). A layer with tied weights requires a "partner" layer -- the
    tied layer borrows the weights from its partner and uses the transpose of
    them to perform its feedforward mapping. Thus, tied layers do not have their
    own weights. On the other hand, tied layers do have their own bias values,
    but these can be fixed to zero during learning to simulate networks with no
    bias (e.g., PCA on mean-centered data).

    Attributes
    ----------
    partner : :class:`Layer`
        The "partner" layer to which this layer is tied.
    '''

    def __init__(self, partner, **kwargs):
        self.partner = partner
        kwargs['inputs'] = partner.size
        kwargs['size'] = list(partner.inputs.values())[0]
        kwargs['name'] = 'tied-{}'.format(partner.name)
        super(Tied, self).__init__(**kwargs)

    def transform(self, inputs):
        '''Transform the inputs for this layer into an output for the layer.

        Parameters
        ----------
        inputs : dict of Theano expressions
            Symbolic inputs to this layer, given as a dictionary mapping string
            names to Theano expressions. See :func:`Layer.connect`.

        Returns
        -------
        outputs : dict of Theano expressions
            A map from string output names to Theano expressions for the outputs
            from this layer. This layer type generates a "pre" output that gives
            the unit activity before applying the layer's activation function,
            and an "out" output that gives the post-activation output.
        updates : list of update pairs
            An empty sequence of updates.
        '''
        x = self._only_input(inputs)
        pre = TT.dot(x, self.partner.find('w').T) + self.find('b')
        return dict(pre=pre, out=self.activate(pre)), []

    def setup(self):
        '''Set up the parameters and initial values for this layer.'''
        # this layer does not create a weight matrix!
        self.add_bias('b', self.size)

    def to_spec(self):
        '''Create a specification dictionary for this layer.

        Returns
        -------
        spec : dict
            A dictionary specifying the configuration of this layer.
        '''
        spec = super(Tied, self).to_spec()
        spec['partner'] = self.partner.name
        return spec


class Maxout(Layer):
    r'''A maxout layer computes a piecewise linear activation function.

    '''

    def __init__(self, **kwargs):
        self.pieces = kwargs.pop('pieces')
        super(Maxout, self).__init__(**kwargs)

    def setup(self):
        '''Set up the parameters and initial values for this layer.'''
        self.add_weights('w')
        self.add_bias('b', self.size)

    def log(self):
        '''Log some information about this layer.'''
        logging.info('layer %s: %s -> %s (x%s), %s, %d parameters',
                     self.name,
                     self.input_size,
                     self.size,
                     self.pieces,
                     self.activate.__theanets_name__,
                     self.num_params)

    def transform(self, inputs):
        '''Transform the inputs for this layer into an output for the layer.

        Parameters
        ----------
        inputs : dict of Theano expressions
            Symbolic inputs to this layer, given as a dictionary mapping string
            names to Theano expressions. See :func:`Layer.connect`.

        Returns
        -------
        outputs : dict of Theano expressions
            A map from string output names to Theano expressions for the outputs
            from this layer. This layer type generates a "pre" output that gives
            the unit activity before applying the layer's activation function,
            and an "out" output that gives the post-activation output.
        updates : list of update pairs
            An empty sequence of state updates.
        '''
        x = self._only_input(inputs)
        pre = TT.dot(x, self.find('w')).max(axis=2) + self.find('b')
        return dict(pre=pre, out=self.activate(pre)), []

    def add_weights(self, name, mean=0, std=None, sparsity=0):
        '''Helper method to create a new weight matrix.

        Parameters
        ----------
        name : str
            Name of the parameter to add.
        mean : float, optional
            Mean value for randomly-initialized weights. Defaults to 0.
        std : float, optional
            Standard deviation of initial matrix values. Defaults to
            :math:`1 / sqrt(n_i + n_o)`.
        sparsity : float, optional
            Fraction of weights to set to zero. Defaults to 0.
        '''
        nin = self.input_size
        nout = self.size
        std = std or 1 / np.sqrt(nin + nout)
        p = self.kwargs.get('sparsity_{}'.format(name),
                            self.kwargs.get('sparsity', sparsity))
        def rm():
            return random_matrix(nin, nout, mean, std, sparsity=p)[:, :, None]
        # stack up weight matrices for the pieces in our maxout.
        arr = np.concatenate([rm() for _ in range(self.pieces)], axis=2)
        self.params.append(theano.shared(arr, name=self._fmt(name)))

    def to_spec(self):
        '''Create a specification dictionary for this layer.

        Returns
        -------
        spec : dict
            A dictionary specifying the configuration of this layer.
        '''
        spec = super(Maxout, self).to_spec()
        spec['pieces'] = self.pieces
        return spec

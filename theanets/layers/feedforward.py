# -*- coding: utf-8 -*-

r'''Feedforward layers for neural network computation graphs.'''

from __future__ import division

import climate
import numpy as np
import theano.sparse as SS
import theano.tensor as TT

from . import base
from .. import util

logging = climate.get_logger(__name__)

__all__ = [
    'Classifier',
    'Feedforward',
    'Tied',
]


class Feedforward(base.Layer):
    '''A feedforward neural network layer performs a transform of its input.

    More precisely, feedforward layers as implemented here perform an affine
    transformation of their input, followed by a potentially nonlinear
    :ref:`activation function <activations>` performed elementwise on the
    transformed input.

    Feedforward layers are the fundamental building block on which most neural
    network models are built.

    Notes
    -----

    This layer can be constructed using the forms ``'feedforward'`` or ``'ff'``.

    *Parameters*

    - With one input:

      - ``b`` --- bias
      - ``w`` --- weights

    - With :math:`N>1` inputs:

      - ``b`` --- bias
      - ``w_1`` --- weight for input 1
      - ``w_2`` ...
      - ``w_N`` --- weight for input :math:`N`

    *Outputs*

    - ``out`` --- the post-activation state of the layer
    - ``pre`` --- the pre-activation state of the layer
    '''

    __extra_registration_keys__ = ['ff']

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
        for name, layer in self._resolved_inputs.items():
            label = 'w' if len(self.inputs) == 1 else 'w_{}'.format(name)
            self.add_weights(label, layer.size, self.size)
        self.add_bias('b', self.size)


class Classifier(Feedforward):
    '''A classifier layer performs a softmax over a linear input transform.

    Classifier layers are typically the "output" layer of a classifier network.

    This layer type really only wraps the output activation of a standard
    :class:`Feedforward` layer.

    Notes
    -----

    The classifier layer is just a vanilla :class:`Feedforward` layer that uses
    a ``'softmax'`` output :ref:`activation <activations>`.
    '''

    __extra_registration_keys__ = ['softmax']

    def __init__(self, **kwargs):
        kwargs['activation'] = 'softmax'
        super(Classifier, self).__init__(**kwargs)


class Tied(base.Layer):
    '''A tied-weights feedforward layer shadows weights from another layer.

    Notes
    -----

    Tied weights are typically featured in some types of autoencoder models
    (e.g., PCA). A layer with tied weights requires a "partner" layer -- the
    tied layer borrows the weights from its partner and uses the transpose of
    them to perform its feedforward mapping. Thus, tied layers do not have their
    own weights. On the other hand, tied layers do have their own bias values,
    but these can be fixed to zero during learning to simulate networks with no
    bias (e.g., PCA on mean-centered data).

    *Parameters*

    - ``b`` --- bias

    *Outputs*

    - ``out`` --- the post-activation state of the layer
    - ``pre`` --- the pre-activation state of the layer

    Parameters
    ----------
    partner : str or :class:`theanets.layers.base.Layer`
        The "partner" layer to which this layer is tied.

    Attributes
    ----------
    partner : :class:`theanets.layers.base.Layer`
        The "partner" layer to which this layer is tied.
    '''

    def __init__(self, partner, **kwargs):
        self.partner = partner
        kwargs['size'] = None
        if isinstance(partner, base.Layer):
            kwargs['size'] = partner.input_size
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

    def resolve(self, layers):
        super(Tied, self).resolve(layers)

        if isinstance(self.partner, util.basestring):
            # if the partner is named, just get that layer.
            matches = [l for l in layers if l.name == self.partner]
            if len(matches) != 1:
                raise util.ConfigurationError(
                    'layer "{}": cannot find partner "{}"'.format(
                        self.name, partner))
            self.partner = matches[0]

        self.size = self.partner.input_size

    def setup(self):
        '''Set up the parameters and initial values for this layer.'''
        # this layer does not create a weight matrix!
        self.add_bias('b', self.size)

    def log(self):
        '''Log some information about this layer.'''
        inputs = ', '.join('({0}){1.size}'.format(n, l)
                           for n, l in self._resolved_inputs.items())
        logging.info('layer %s "%s" << "%s": %s -> %s, %s, %d parameters',
                     self.__class__.__name__,
                     self.name,
                     self.partner.name,
                     inputs,
                     self.size,
                     getattr(self.activate, 'name', self.activate),
                     sum(np.prod(p.get_value().shape) for p in self.params))

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

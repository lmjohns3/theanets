# -*- coding: utf-8 -*-

'''Convolutional layers "scan" over input data.'''

from __future__ import division

import numpy as np
import theano
import theano.tensor as TT

from . import base
from .. import util

__all__ = [
    'Conv1',
    'Conv2',
    'Pool1',
    'Pool2',
]


class Convolution(base.Layer):
    '''Convolution layers convolve filters over the input arrays.

    Parameters
    ----------
    filter_size : (int, int)
        Size of the convolution filters for this layer.
    stride : (int, int), optional
        Apply convolutions with this stride; i.e., skip this many samples
        between convolutions. Defaults to (1, 1)---that is, no skipping.
    border_mode : str, optional
        Compute convolutions with this border mode. Defaults to 'valid'.
    '''

    def __init__(self, filter_size, stride=(1, 1), border_mode='valid', **kwargs):
        self.filter_size = filter_size
        self.stride = stride
        self.border_mode = border_mode
        super(Convolution, self).__init__(**kwargs)

    def log(self):
        inputs = ', '.join('"{0}" {1}'.format(*ns) for ns in self._input_shapes.items())
        util.log('layer {0.__class__.__name__} "{0.name}" '
                 '{0.output_shape} {1} {0.border_mode} '
                 'filters {2}{3} from {4}', self,
                 getattr(self.activate, 'name', self.activate),
                 'x'.join(str(i) for i in self.filter_size),
                 ''.join('+{}'.format(i) for i in self.stride),
                 inputs)
        util.log('learnable parameters: {}', self.log_params())

    def add_conv_weights(self, name, mean=0, std=None, sparsity=0):
        '''Add a convolutional weight array to this layer's parameters.

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
        nout = self.output_size
        mean = self.kwargs.get(
            'mean_{}'.format(name),
            self.kwargs.get('mean', mean))
        std = self.kwargs.get(
            'std_{}'.format(name),
            self.kwargs.get('std', std or 1 / np.sqrt(nin + nout)))
        sparsity = self.kwargs.get(
            'sparsity_{}'.format(name),
            self.kwargs.get('sparsity', sparsity))
        arr = np.zeros((nout, nin) + self.filter_size, util.FLOAT)
        for r in range(self.filter_size[0]):
            for c in range(self.filter_size[1]):
                arr[:, :, r, c] = util.random_matrix(
                    nout, nin, mean, std, sparsity=sparsity, rng=self.rng)
        self._params.append(theano.shared(arr, name=self._fmt(name)))


class Conv1(Convolution):
    '''1-dimensional convolutions run over one data axis.

    Notes
    -----

    One-dimensional convolution layers are typically used in ``theanets`` models
    that use recurrent inputs and outputs, i.e.,
    :class:`theanets.recurrent.Autoencoder`,
    :class:`theanets.recurrent.Predictor`,
    :class:`theanets.recurrent.Classifier`, or
    :class:`theanets.recurrent.Regressor`.

    The convolution will be applied over the "time" dimension (axis 1).

    Parameters
    ----------
    filter_size : int
        Length of the convolution filters for this layer.
    stride : int, optional
        Apply convolutions with this stride; i.e., skip this many samples
        between convolutions. Defaults to 1, i.e., no skipping.
    border_mode : str, optional
        Compute convolutions with this border mode. Defaults to 'valid'.
    '''

    def __init__(self, filter_size, stride=1, border_mode='valid', **kwargs):
        super(Conv1, self).__init__(
            filter_size=(1, filter_size),
            stride=(1, stride),
            border_mode=border_mode,
            **kwargs)

    def setup(self):
        self.add_conv_weights('w')
        self.add_bias('b', self.output_size)

    def resolve_outputs(self):
        if self.input_shape is None or self.input_shape[0] is None:
            return super(Conv1, self).resolve_outputs()
        image = np.array(self.input_shape[:-1])
        kernel = np.array(self.filter_size)
        result = image
        if self.border_mode == 'full':
            result = image + kernel - 1
        if self.border_mode == 'valid':
            result = image - kernel + 1
        self._output_shapes['out'] = tuple(result) + (self.kwargs['size'], )

    def transform(self, inputs):
        # input is:     (batch, time, input)
        # conv2d wants: (batch, input, 1, time)
        x = inputs[self.input_name].dimshuffle(0, 2, 'x', 1)

        pre = TT.nnet.conv2d(
            x,
            self.find('w'),
            image_shape=(None, self.input_size, 1, None),
            filter_shape=(self.output_size, self.input_size) + self.filter_size,
            border_mode=self.border_mode,
            subsample=self.stride,
        ).dimshuffle(0, 3, 1, 2)[:, :, :, 0] + self.find('b')
        # conv2d output is: (batch, output, 1, time)
        # we want:          (batch, time, output)
        # (have to do [:, :, :, 0] to remove unused trailing dimension)

        return dict(pre=pre, out=self.activate(pre)), []


class Conv2(Convolution):
    '''2-dimensional convolutions run over two data axes.

    Two-dimensional convolution layers are standard image processing techniques.
    In theanets, these layers expect an input consisting of (num-examples,
    width, height, num-channels).

    Parameters
    ----------
    filter_size : (int, int)
        Size of the convolution filters for this layer.
    stride : (int, int), optional
        Apply convolutions with this stride; i.e., skip this many samples
        between convolutions. Defaults to (1, 1), i.e., no skipping.
    border_mode : str, optional
        Compute convolutions with this border mode. Defaults to 'valid'.
    '''

    def setup(self):
        self.add_conv_weights('w')
        self.add_bias('b', self.output_size)

    def resolve_outputs(self):
        shape = self.input_shape
        if shape is None or shape[0] is None or shape[1] is None:
            return super(Conv2, self).resolve_outputs()
        image = np.array(shape[:-1])
        kernel = np.array(self.filter_size)
        result = image
        if self.border_mode == 'full':
            result = image + kernel - 1
        if self.border_mode == 'valid':
            result = image - kernel + 1
        self._output_shapes['out'] = tuple(result) + (self.kwargs['size'], )

    def transform(self, inputs):
        # input is:     (batch, width, height, input)
        # conv2d wants: (batch, input, width, height)
        x = inputs[self.input_name].dimshuffle(0, 3, 1, 2)

        pre = TT.nnet.conv2d(
            x,
            self.find('w'),
            image_shape=(None, self.input_size, None, None),
            filter_shape=(self.output_size, self.input_size) + self.filter_size,
            border_mode=self.border_mode,
            subsample=self.stride,
        ).dimshuffle(0, 2, 3, 1) + self.find('b')
        # conv2d output is: (batch, output, width, height)
        # we want:          (batch, width, height, output)

        return dict(pre=pre, out=self.activate(pre)), []


class Pooling(base.Layer):
    '''
    '''


class Pool1(Pooling):
    '''
    '''

    def transform(self, inputs):
        # input is:     (batch, time, input)
        # conv2d wants: (batch, input, time, 1)
        x = inputs[self.input_name].dimshuffle(0, 2, 1, 'x')

        pre = TT.signal.downsample.max_pool_2d(
            x, self.pool_size, st=self.stride, mode=self.mode,
        ).dimshuffle(0, 2, 1, 3)[:, :, :, 0]
        # conv2d output is: (batch, output, time, 1)
        # we want:          (batch, time, output)

        return dict(pre=pre, out=self.activate(pre)), []


class Pool2(Pooling):
    '''
    '''

    def transform(self, inputs):
        # input is:     (batch, width, height, input)
        # conv2d wants: (batch, input, width, height)
        x = inputs[self.input_name].dimshuffle(0, 3, 1, 2)

        pre = TT.signal.downsample.max_pool_2d(
            x, self.pool_size, st=self.stride, mode=self.mode,
        ).dimshuffle(0, 2, 3, 1)
        # conv2d output is: (batch, output, width, height)
        # we want:          (batch, width, height, output)

        return dict(pre=pre, out=self.activate(pre)), []

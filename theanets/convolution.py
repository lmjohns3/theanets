# -*- coding: utf-8 -*-

'''This module contains convolution network structures.'''

from . import feedforward


class Regressor(feedforward.Regressor):
    '''A regressor attempts to produce a target output.

    A convolutional regression model takes the following inputs during training:

    - ``x``: A three-dimensional array of input data. Each element of axis 0 of
      ``x`` is expected to be one moment in time. Each element of axis 1 of
      ``x`` holds a single sample from a batch of data. Each element of axis 2
      of ``x`` represents the measurements of a particular input variable across
      all times and all data items.

    - ``targets``: A two-dimensional array of target output data. Each element
      of axis 0 of ``targets`` is expected to be one moment in time. Each
      element of axis 1 of ``targets`` holds a single sample from a batch of
      data. Each element of axis 2 of ``targets`` represents the measurements of
      a particular output variable across all times and all data items.
    '''

    def __init__(self, layers=(), loss='mse', weighted=False):
        super(feedforward.Regressor, self).__init__(
            layers=layers, loss=loss, in_dim=4, out_dim=2, weighted=weighted)


class Classifier(feedforward.Classifier):
    '''A classifier attempts to match a 1-hot target output.

    Unlike a feedforward classifier, where the target labels are provided as a
    single vector, a recurrent classifier requires a vector of target labels for
    each time step in the input data. So a recurrent classifier model requires
    the following inputs for training:

    - ``x``: A three-dimensional array of input data. Each element of axis 0 of
      ``x`` is expected to be one moment in time. Each element of axis 1 of
      ``x`` holds a single sample in a batch of data. Each element of axis 2 of
      ``x`` represents the measurements of a particular input variable across
      all times and all data items in a batch.

    - ``labels``: A one-dimensional vector of integer target labels. Each
      element of ``labels`` is expected to be the class index for a single batch
      item.
    '''

    def __init__(self, layers=(), loss='xe', weighted=False):
        super(feedforward.Classifier, self).__init__(
            layers=layers, loss=loss, in_dim=4, out_dim=1, weighted=weighted)

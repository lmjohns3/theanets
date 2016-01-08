# -*- coding: utf-8 -*-

'''This module contains convolution network structures.'''

from . import feedforward


class Regressor(feedforward.Regressor):
    '''A regressor attempts to produce a target output.'''

    INPUT_NDIM = 4
    '''Number of dimensions for holding input data arrays.'''


class Classifier(feedforward.Classifier):
    '''A classifier attempts to match a 1-hot target output.'''

    INPUT_NDIM = 4
    '''Number of dimensions for holding input data arrays.'''

#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Example using the theanets package for learning the XOR relation.'''

import numpy as np
import theanets

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype='f')
Y = np.array([[0], [1], [1], [0]], dtype='f')

net = theanets.Regressor([dict(size=2, input_noise=0.3), 2, 1])
net.train([X, Y], algo='rmsprop', patience=10, batch_size=4)

theanets.log('Input: {}', [list(x) for x in X])
theanets.log('XOR output: {}', Y.T)
theanets.log('NN XOR predictions: {}', net.predict(X.astype('f')).T.round(2))

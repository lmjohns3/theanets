#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Example using the theanets package for learning the XOR relation.'''

import climate
import logging
import numpy as np
import theanets

climate.enable_default_logging()

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype='f')
Y = np.array([[0], [1], [1], [0]], dtype='f')

e = theanets.Experiment(theanets.Regressor, layers=(2, 2, 1))
e.train([X, Y], optimize='rprop', min_improvement=0.2, patience=500)

logging.info("Input:\n%s", X)
logging.info("XOR output:\n%s", Y)
logging.info("NN XOR predictions:\n%s", e.network(X.astype('f')).round(2))

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

net = theanets.Regressor([2, 2, 1])
net.train([X, Y], algo='rprop', patience=10, batch_size=4)

logging.info("Input:\n%s", X)
logging.info("XOR output:\n%s", Y)
logging.info("NN XOR predictions:\n%s", net.predict(X.astype('f')).round(2))

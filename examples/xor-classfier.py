#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cPickle
import lmj.cli
import matplotlib.pyplot as plt
import numpy as np
import theano
import theanets


"""
This is an example of using the theanets package for learning an XOR classifier.
"""

lmj.cli.enable_default_logging()

X = np.array([
               [0.0, 0.0],
               [0.0, 1.0],
               [1.0, 0.0],
               [1.0, 1.0],
             ])

Y = np.array([0, 1, 1, 0, ])

train = [X,  Y.astype('int32')]

e = theanets.Experiment(theanets.Classifier,
                        layers=(2, 5, 2),
                        activation = 'tanh',
#                        learning_rate=.005,
#                        learning_rate_decay=.1,
#                        patience=20,
                        optimize="sgd",
                        num_updates=10,
#                        tied_weights=True,
#                        batch_size=32,
                        )
e.run(train, train)

print e.network.predict(X)



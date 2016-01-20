#!/usr/bin/env python

import climate
import logging
import matplotlib.pyplot as plt
import numpy as np
import numpy.random as rng
import theanets

climate.enable_default_logging()

BATCH_SIZE = 32
STEPS = 20

weight = np.zeros((STEPS, BATCH_SIZE, 1), 'f')
weight[-1:] = 1


def examples():
    x, z = rng.uniform(0, 1, size=(2, STEPS, BATCH_SIZE, 1))
    y = np.zeros((STEPS, BATCH_SIZE, 1))
    idx = list(range(STEPS - 1))
    for b in range(BATCH_SIZE):
        rng.shuffle(idx)
        y[idx[0], b] = 1
        y[idx[1], b] = 1
        z[-1, b] = x[idx[0], b] + x[idx[1], b]
    return np.concatenate([x, y], axis=2).astype('f'), z.astype('f'), weight

src, tgt, wgt = examples()
logging.info('data batches: %s -> %s @ %s', src.shape, tgt.shape, wgt.shape)

e = theanets.Experiment(
    theanets.recurrent.Regressor,
    layers=(2, dict(form='rnn', activation='relu', size=100, radius=1), 1),
    weighted=True)
e.train(examples)
prd = e.network.transform(src)

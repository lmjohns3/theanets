#!/usr/bin/env python

import numpy.random as rng
import theanets

TIME = 10
BATCH_SIZE = 32

e = theanets.Experiment(
    theanets.recurrent.Autoencoder,
    layers=(3, ('rnn', 10), 3),
    batch_size=BATCH_SIZE)


def generate():
    return [rng.randn(TIME, BATCH_SIZE, 3).astype('f')]

batch = generate()
theanets.log('data batches: {}', batch[0].shape)

e.train(generate)

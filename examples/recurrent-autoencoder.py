#!/usr/bin/env python

import climate
import logging
import numpy.random as rng
import theanets

climate.enable_default_logging()

TIME = 10
BATCH_SIZE = 32

e = theanets.Experiment(
    theanets.recurrent.Autoencoder,
    layers=(3, ('rnn', 10), 3),
    batch_size=BATCH_SIZE)


def generate():
    return [rng.randn(TIME, BATCH_SIZE, 3).astype('f')]

batch = generate()
logging.info('data batches: %s', batch[0].shape)

e.train(generate)

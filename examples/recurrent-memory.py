#!/usr/bin/env python

import climate
import logging
import matplotlib.pyplot as plt
import numpy as np
import theanets

climate.enable_default_logging()

TIME = 10
BITS = 3
BATCH_SIZE = 32

mask = np.ones((TIME, BATCH_SIZE, 1), bool)
mask[:TIME - BITS] = 0

e = theanets.Experiment(
    theanets.recurrent.Regressor,
    layers=(1, ('rnn', 10), 1),
    weighted=True)

def generate():
    s, t = np.random.randn(2, TIME, BATCH_SIZE, 1).astype('f')
    s[:BITS] = t[-BITS:] = np.random.randn(BITS, BATCH_SIZE, 1)
    return s, t, mask

src, tgt, msk = generate()
logging.info('data batches: %s -> %s @ %s', src.shape, tgt.shape, msk.shape)

e.train(generate, batch_size=BATCH_SIZE)

predict = e.network.predict(src)[:, :, 0]
vm = max(abs(src[:BITS]).max(), abs(predict[-BITS]).max())

def plot(n, z, label):
    ax = plt.subplot(2, 1, n)
    ax.set_frame_on(False)
    for loc, spine in ax.spines.items():
        spine.set_color('none')
    ax.imshow(z, cmap='gray', vmin=-vm, vmax=vm)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel('Example')
    ax.set_ylabel('{} Time'.format(label))

plot(1, src[:, :, 0], 'Source')
plot(2, predict, 'Prediction')

plt.show()

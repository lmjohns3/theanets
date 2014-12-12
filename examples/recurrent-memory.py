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

e = theanets.Experiment(
    theanets.recurrent.Regressor,
    layers=(1, 100, 1),
    recurrent_error_start=TIME - BITS,
    batch_size=BATCH_SIZE)

def generate():
    s = np.zeros((TIME, BATCH_SIZE, 1), 'f')
    t = np.zeros((TIME, BATCH_SIZE, 1), 'f')
    s[:BITS] = t[-BITS:] = np.random.randn(BITS, BATCH_SIZE, 1)
    return [s, t]

src, tgt = generate()
logging.info('data batches: %s -> %s', src.shape, tgt.shape)

e.train(generate)

target = tgt[-BITS:, :, 0]
predict = e.network.predict(src)[-BITS:, :, 0]
vm = max(abs(target).max(), abs(predict).max())

def plot(n, z, label):
    ax = plt.subplot(2, 1, n)
    ax.set_frame_on(False)
    for loc, spine in ax.spines.items():
        spine.set_color('none')
    ax.imshow(z, cmap='gray', vmin=-vm, vmax=vm)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_ylabel(label)

plot(1, target, 'Target')
plot(2, predict, 'Prediction')

plt.show()

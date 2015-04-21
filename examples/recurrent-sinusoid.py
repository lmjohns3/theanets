#!/usr/bin/env python

import climate
import logging
import matplotlib.pyplot as plt
import numpy as np
import numpy.random as rng
import theanets

climate.enable_default_logging()

BATCH_SIZE = 32
T = np.linspace(0, 1, 100)

concat = lambda a, b: np.concatenate([a, b], axis=2).astype('f')

def sines():
    f = np.exp(rng.uniform(0, 1, size=BATCH_SIZE))
    time = np.outer(T, np.ones(BATCH_SIZE))[..., None]
    freq = np.outer(np.ones_like(T), f)[..., None]
    z = 2 * np.pi * freq * time
    return concat(time, freq), concat(np.sin(z), np.cos(z))

src, tgt = sines()
logging.info('data batches: %s -> %s', src.shape, tgt.shape)

e = theanets.Experiment(
    theanets.recurrent.Regressor,
    layers=(2, ('rnn', 10), 2))
e.train(sines, batch_size=BATCH_SIZE)
prd = e.network.predict(src)

for i in range(3):
    ax = plt.subplot(3, 1, i + 1)
    ax.plot(tgt[:, i, 0].flatten(), '--', color='#1f77b4', label='Target Sine')
    ax.plot(tgt[:, i, 1].flatten(), '--', color='#2ca02c', label='Target Cosine')
    ax.plot(prd[:, i, 0].flatten(), '-', color='#1f77b4', lw=2, label='Predicted Sine')
    ax.plot(prd[:, i, 1].flatten(), '-', color='#2ca02c', lw=2, label='Predicted Cosine')
    ax.xaxis.tick_bottom()
    ax.yaxis.tick_left()
    ax.spines['top'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.set_ylabel('Amplitude')
    if i == 0:
        plt.legend()
    if i == 2:
        ax.set_xlabel('Time')
    else:
        ax.set_xticks([])
plt.show()

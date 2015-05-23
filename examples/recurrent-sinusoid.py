#!/usr/bin/env python

import climate
import logging
import matplotlib.pyplot as plt
import numpy as np
import theanets

climate.enable_default_logging()

TAU = 2 * np.pi

BATCH_SIZE = 2
T = np.linspace(0, TAU, 256)

COEFFS = ((2, 1.5), (3, 1.8), (4, 1.1))

SIN = sum(c * np.sin(TAU * f * T) for c, f in COEFFS)
COS = sum(c * np.cos(TAU * f * T) for c, f in COEFFS)
W = np.concatenate([SIN[:, None], COS[:, None]], axis=1)
WAVES = np.concatenate([W[:, None, :]] * BATCH_SIZE, axis=1).astype('f')
ZERO = np.zeros((len(T), BATCH_SIZE, 1), 'f')

e = theanets.Experiment(
    theanets.recurrent.Regressor,
    layers=(1, dict(form='clockwork', size=64, periods=(1, 4, 16, 64)), 2))

e.train([ZERO, WAVES], batch_size=BATCH_SIZE, learning_rate=0.001)

prd = e.network.predict(ZERO)
ax = plt.subplot(111)
ax.plot(T, SIN, ':', color='#1f77b4', lw=2, label='Target Sine')
ax.plot(T, COS, ':', color='#2ca02c', lw=2, label='Target Cosine')
ax.plot(T, prd[:, 0, 0].flatten(), '-', color='#1f77b4', lw=2, label='Predicted Sine')
ax.plot(T, prd[:, 0, 1].flatten(), '-', color='#2ca02c', lw=2, label='Predicted Cosine')
ax.xaxis.tick_bottom()
ax.yaxis.tick_left()
ax.spines['top'].set_color('none')
ax.spines['right'].set_color('none')
ax.set_ylabel('Amplitude')
ax.set_xlabel('Time')
plt.legend()
plt.show()

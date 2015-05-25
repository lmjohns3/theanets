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
WAVES = np.concatenate([SIN[:, None, None]] * BATCH_SIZE, axis=1).astype('f')
ZERO = np.zeros((len(T), BATCH_SIZE, 1), 'f')

ax = plt.subplot(111)
ax.plot(T, SIN, ':', label='Target', alpha=0.7)

for name, hidden in (
        ('RNN', ('rnn', 'relu', 64)),
        ('LSTM', ('lstm', 'tanh', 64)),
        ('Clockwork', dict(form='clockwork',
                           activation='linear',
                           size=64,
                           periods=(1, 4, 16, 64))),
):
    logging.info('training %s model', name)
    e = theanets.Experiment(theanets.recurrent.Regressor, layers=(1, hidden, 1))
    e.train([ZERO, WAVES], batch_size=BATCH_SIZE, learning_rate=0.0001, patience=0)
    prd = e.network.predict(ZERO)
    ax.plot(T, prd[:, 0, 0].flatten(), label=name, alpha=0.7)

ax.xaxis.tick_bottom()
ax.yaxis.tick_left()
ax.spines['top'].set_color('none')
ax.spines['right'].set_color('none')
ax.set_ylabel('Amplitude')
ax.set_xlabel('Time')
plt.legend()
plt.show()

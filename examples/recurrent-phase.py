#!/usr/bin/env python

import logging
import numpy as np
import numpy.random as rng
import lmj.cli
import lmj.nn

from matplotlib import pyplot as plt

lmj.cli.enable_default_logging()

e = lmj.nn.Experiment(
    lmj.nn.recurrent.Autoencoder,
    layers=(1, 10, 1), num_updates=20, train_batches=64)

T = 256
K = int(0.5 * T)
S = np.linspace(0, 4 * np.pi, T)

def sines(i=0):
    return (0.7 * np.sin(S) + 0.3 * np.sin(i * S / 2)).reshape((T, 1))

e.run(lambda _: [sines(rng.randint(K, T))],
      lambda _: [sines(rng.randint(0, K))])

source = sines(13)
match = e.network(source)

# plot the input, output, and error of the network.

t = np.arange(T)

ax = plt.subplot(111)
ax.xaxis.tick_bottom()
ax.yaxis.tick_left()
for loc, spine in ax.spines.iteritems():
    if loc in 'left bottom':
        spine.set_position(('outward', 6))
    elif loc in 'right top':
        spine.set_color('none')

ax.plot(t, source, '.-', c='#111111', label='Target')
ax.plot(t, match, '.-', c='#1f77b4', label='Output')
ax.plot(t, abs(source - match), '.-', c='#d62728', label='Error')

ax.set_xlim(0, T)

plt.legend()
plt.show()

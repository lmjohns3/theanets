#!/usr/bin/env python

import logging
import numpy as np
import lmj.cli
import lmj.nn

from matplotlib import pyplot as plt

lmj.cli.enable_default_logging()

T = 256
S = np.linspace(0, 4 * np.pi, T)

def sines(i=0):
    return (0.7 * np.sin(S) + 0.3 * np.sin(i * S / 2)).reshape((T, 1))


class Main(lmj.nn.Main):
    def get_network(self):
        return lmj.nn.recurrent.Autoencoder

    def get_datasets(self):
        train = np.array([sines(i) for i in range(64, 256)])
        dev = np.array([sines(i) for i in range(64)])
        return train, dev

m = Main(layers=(1, 10, 1), batch_size=1, num_updates=10, learning_rate=0.01)
m.train()

source = sines(13)
match = m.net(source)

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

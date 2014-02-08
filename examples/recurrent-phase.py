#!/usr/bin/env python

import climate
import matplotlib.pyplot as plt
import numpy as np
import numpy.random as rng
import theanets

climate.enable_default_logging()

T = 256
K = int(0.5 * T)
S = np.linspace(0, 4 * np.pi, T)

def sines(i=0):
    return (0.7 * np.sin(S) + 0.3 * np.sin(i * S / 2)).reshape((T, 1)).astype('f')

# set up a network and train it using some sinusoidal data.

e = theanets.Experiment(
    theanets.recurrent.Autoencoder,
    layers=(1, 10, 1),
    num_updates=20,
    train_batches=64)

e.run(lambda: [sines(rng.randint(K, T))],
      lambda: [sines(rng.randint(0, K))])

# plot the input, output, and error of the network.

ax = plt.subplot(111)
ax.xaxis.tick_bottom()
ax.yaxis.tick_left()
for loc, spine in ax.spines.iteritems():
    if loc in 'left bottom':
        spine.set_position(('outward', 6))
    elif loc in 'right top':
        spine.set_color('none')

source = sines(13)
match = e.network.predict(source)
ax.plot(source, '.-', c='#111111', label='Target')
ax.plot(match, '.-', c='#1f77b4', label='Output')
ax.plot(abs(source - match), '.-', c='#d62728', label='Error')

ax.set_xlim(0, T)
ax.set_xlabel('Time')
ax.set_ylabel('Amplitude')

plt.legend()
plt.show()

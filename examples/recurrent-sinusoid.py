#!/usr/bin/env python

import climate
import matplotlib.pyplot as plt
import numpy as np
import numpy.random as rng
import theanets

climate.enable_default_logging()

S = np.linspace(0, 4 * np.pi, 256)

def wave(i=0):
    return (0.4 * np.sin(S) + 0.3 * np.sin(i * S / 2))[:, None, None]

def waves(n=64):
    return np.concatenate([wave(rng.randint(15, 30)) for _ in range(n)], axis=1).astype('f')

# set up a network and train it using some sinusoidal data.

e = theanets.Experiment(theanets.recurrent.Regressor, layers=(2, 10, 1))

def sum_waves():
    x = waves()
    y = waves()
    return [np.concatenate([x, y], axis=2), x + y]

e.train(sum_waves, batch_size=64, train_batches=16)

# use the network to predict a novel output.

batch = np.zeros((len(S), e.args.batch_size, 2)).astype('f')
batch[:, :1, 0:1] = wave(13).astype('f')
batch[:, :1, 1:2] = wave(14).astype('f')

predict = e.network.predict(batch)[:, 0, :]

# plot the input, output, and error of the network.

def plot(x, label):
    ax.plot(x.flatten(), '.-', label=label)

ax = plt.subplot(111)
ax.xaxis.tick_bottom()
ax.yaxis.tick_left()
for loc, spine in ax.spines.items():
    if loc in 'left bottom':
        spine.set_position(('outward', 6))
    elif loc in 'right top':
        spine.set_color('none')

plot(batch[:, 0, 0] + batch[:, 0, 1], 'Target')
plot(predict, 'Prediction')

ax.set_xlim(0, len(S))
ax.set_xlabel('Time')
ax.set_ylabel('Amplitude')

plt.legend()
plt.show()

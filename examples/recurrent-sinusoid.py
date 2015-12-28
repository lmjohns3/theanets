#!/usr/bin/env python

'''This example compares recurrent layer performance on a sine-generation task.

The task is to generate a complex sine wave that is constructed as a
superposition of a small set of pure frequencies. All networks are constructed
with one input (which receives all zero values), one recurrent hidden layer, and
one output (which is tasked with matching the target sine wave). Each model is
trained and then its predicted output is plotted for easy visual comparison of
the behavior of the different layer models.

For this task, the clockwork RNN layer tends to perform the best of the layer
models, even though the clockwork layer uses the simplest activation (linear)
and has the fewest parameters (~2000 for a 64-node hidden layer, versus ~4000
for a vanilla RNN and ~17000 for an LSTM). The vanilla RNN layer tends to do the
worst, or at the least is the most sensitive to the initialization of the
parameters. The other layer models fall somewhere in the middle but tend only to
match the dominant frequency in the target wave.
'''

import climate
import logging
import matplotlib.pyplot as plt
import numpy as np
import theanets

climate.enable_default_logging()

COLORS = ['#d62728', '#1f77b4', '#2ca02c', '#9467bd', '#ff7f0e',
          '#e377c2', '#8c564b', '#bcbd22', '#7f7f7f', '#17becf']

BATCH_SIZE = 2


# Construct a complex sine wave as a sum of pure-frequency waves.
TAU = 2 * np.pi
T = np.linspace(0, TAU, 256)
SIN = sum(c * np.sin(TAU * f * T) for c, f in ((2, 1.5), (3, 1.8), (4, 1.1)))


# Create an input dataset consisting of all zeros, and an output dataset
# containing the target sine wave. We have to stack the target sine wave here
# because recurrent models expect a tensor with three dimensions, and the batch
# size for recurrent networks must be greater than 1.
ZERO = np.zeros((BATCH_SIZE, len(T), 1), 'f')
WAVES = np.concatenate([SIN[None, :, None]] * BATCH_SIZE, axis=0).astype('f')


# Set up plotting axes to show the output result and learning curves.
_, (wave_ax, learn_ax) = plt.subplots(2, 1)

# Plot the target wave.
wave_ax.plot(T, SIN, ':', label='Target', alpha=0.7, color='#111111')


# For each layer type, train a model containing that layer, and plot its
# predicted output.
for i, layer in enumerate((
        dict(form='rnn', activation='relu', diagonal=0.5),
        dict(form='rrnn', activation='relu', rate='vector', diagonal=0.5),
        dict(form='scrn', activation='linear'),
        dict(form='gru', activation='relu'),
        dict(form='lstm', activation='tanh'),
        dict(form='clockwork', activation='linear', periods=(1, 4, 16, 64)))):
    name = layer['form']
    layer['size'] = 64
    logging.info('training %s model', name)
    net = theanets.recurrent.Regressor([1, layer, 1])
    losses = []
    for tm, _ in net.itertrain([ZERO, WAVES],
                               monitor_gradients=True,
                               batch_size=BATCH_SIZE,
                               algorithm='rmsprop',
                               learning_rate=0.0001,
                               momentum=0.9,
                               min_improvement=0.01):
        losses.append(tm['loss'])
    prd = net.predict(ZERO)
    wave_ax.plot(T, prd[0, :, 0].flatten(), label=name, alpha=0.7, color=COLORS[i])
    learn_ax.plot(losses, label=name, alpha=0.7, color=COLORS[i])


# Make the plots look nice.
for ax in [wave_ax, learn_ax]:
    ax.xaxis.tick_bottom()
    ax.yaxis.tick_left()
    ax.spines['top'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.spines['bottom'].set_position(('outward', 6))
    ax.spines['left'].set_position(('outward', 6))

wave_ax.set_ylabel('Amplitude')
wave_ax.set_xlabel('Time')

learn_ax.set_ylabel('Loss')
learn_ax.set_xlabel('Training Epoch')
learn_ax.grid(True)

plt.legend()
plt.show()

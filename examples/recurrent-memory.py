#!/usr/bin/env python

'''This example examines recurrent performance in a memory task.

In the memory task, a network is supposed to read in T frames of n-dimensional
data and reproduce the first t frames at the output of the network, after
reading in T - t additional frames of n-dimensional data.

This task is quite difficult for most neural network models, since the hidden
layer in the network must effectively store the first inputs somewhere, preserve
those values for an arbitrary amount of time (while also discarding any new
inputs), and then reproduce the stored inputs in the proper order.

This example uses a vanilla RNN to perform this task, but the network
configuration can easily be changed to test the performance of different
layer types (or even multilayer architectures).
'''

import climate
import logging
import matplotlib.pyplot as plt
import numpy as np
import theanets

climate.enable_default_logging()

TIME = 10  # Total numer of time steps.
BITS = 3   # Number of steps to remember/reproduce.
BATCH_SIZE = 32


# Here we create a mask that will be used to weight the target outputs for the
# network. These weights are zero everywhere except for the last BITS time
# steps, which forces the network to do anything it can to reproduce the input
# pattern at the end of the output.
mask = np.ones((BATCH_SIZE, TIME, 1), bool)
mask[:, :TIME - BITS - 1] = 0


# We use a callable to generate a batch of random input data to present to our
# network model. Each batch consists of a random input pattern, a random output
# pattern whose final BITS elements correspond to the initial BITS elements of
# the input, and the fixed weight mask from above.
def generate():
    s, t = np.random.randn(2, BATCH_SIZE, TIME, 1).astype('f')
    s[:, :BITS] = t[:, -BITS:] = np.random.randn(BATCH_SIZE, BITS, 1)
    return s, t, mask

src, tgt, msk = generate()
logging.info('data batches: %s -> %s @ %s', src.shape, tgt.shape, msk.shape)


# Create a new recurrent regression model and train it up.
net = theanets.recurrent.Regressor(
    layers=(1, dict(form='rnn', activation='relu', size=10, diagonal=1), 1),
    weighted=True)

net.train(generate,
          batch_size=BATCH_SIZE,
          algorithm='rmsprop',
          max_gradient_norm=1,
          learning_rate=0.001,
          momentum=0.9,
          monitor_gradients=True)


# Now we plot the results. Our plot contains two rows. On the top row, a random
# batch of input values are shown -- time is on the y-axis, and the examples are
# laid out along the x-axis. On the bottom row, the outputs from the network
# model are shown -- again, time and example are on the y- and x-axes,
# respectively.
def plot(n, z, label, rectangle):
    ax = plt.subplot(2, 1, n)
    ax.set_frame_on(False)
    for loc, spine in ax.spines.items():
        spine.set_color('none')
    ax.imshow(z, cmap='gray', vmin=-vm, vmax=vm)
    ax.fill_between([-0.5, BATCH_SIZE - 0.5],
                    rectangle - 0.5,
                    rectangle + BITS - 0.5,
                    lw=0, color='#17becf', alpha=0.3)
    ax.set_xticks([])
    ax.set_yticks([])
    if n == 2:
        ax.set_xlabel('Example')
    ax.set_ylabel(label)

out = net.predict(src)[:, :, 0].T
vm = max(abs(src[:, :BITS]).max(), abs(out[:, -BITS]).max())

plot(1, src[:, :, 0].T, 'Input', 0)
plot(2, out, 'Output', TIME - BITS)

plt.show()

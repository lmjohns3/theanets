#!/usr/bin/env python

import climate
import logging
import matplotlib.pyplot as plt
import numpy.random as rng
import theanets

climate.enable_default_logging()

TIME = 10
BATCH_SIZE = 32

e = theanets.Experiment(
    theanets.recurrent.Autoencoder,
    layers=(3, 100, 3),
    recurrent_error_start=TIME - 1,
    batch_size=BATCH_SIZE)

def generate():
    r = rng.randn(TIME, BATCH_SIZE, 3).astype('f')
    r[-1] = r[0]
    return [r]

batch = generate()
logging.info('data batches: %s', batch[0].shape)

e.train(generate)

target = batch[0][-1]
predict = e.network.predict(batch[0])[-1]
vm = max(abs(target).max(), abs(predict).max())

ax = plt.subplot(211)
ax.set_frame_on(False)
for loc, spine in ax.spines.items():
    spine.set_color('none')
ax.imshow(target.T, cmap='gray', vmin=-vm, vmax=vm)
ax.set_xticks([])
ax.set_yticks([])

ax = plt.subplot(212)
ax.set_frame_on(False)
for loc, spine in ax.spines.items():
    spine.set_color('none')
ax.imshow(predict.T, cmap='gray', vmin=-vm, vmax=vm)
ax.set_yticks([])

plt.show()
